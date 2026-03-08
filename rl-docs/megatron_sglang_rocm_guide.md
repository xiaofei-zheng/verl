# verl Megatron + sglang RL 训练指南（ROCm MI355X）

环境：ROCm MI355X × 8 GPU，verl + Megatron-Core + sglang rollout，单节点

## 目录

1. [依赖安装](#1-依赖安装)
2. [模型准备](#2-模型准备)
3. [代码修改（必须）](#3-代码修改必须)
4. [训练脚本配置](#4-训练脚本配置)
5. [问题排查记录](#5-问题排查记录)
6. [训练结果](#6-训练结果)
7. [已知问题与待解决项](#7-已知问题与待解决项)

---

## 1. 依赖安装

在已有 sglang ROCm 环境的基础上，需要额外安装以下 Megatron 相关依赖：

### 1.1 Megatron-Core

```bash
pip install megatron-core
```

### 1.2 mbridge（HuggingFace ↔ Megatron 权重转换）

```bash
pip install mbridge
```

### 1.3 torch_memory_saver（ROCm 版本需要手动 patch）

```bash
git clone https://github.com/zhuohan123/torch_memory_saver_numa /tmp/torch_memory_saver_numa
cd /tmp/torch_memory_saver_numa
```

编译前需要在 `csrc/torch_memory_saver_hip.cpp` 中添加缺失的头文件：

```cpp
#include <cassert>  // ROCm 编译需要，原代码缺少
```

然后安装：

```bash
pip install .
```

### 1.4 TransformerEngine（ROCm 版本）

```bash
git clone --depth 1 https://github.com/ROCm/TransformerEngine.git
cd TransformerEngine

# 只初始化必要的子模块，跳过 googletest（会卡住）
git submodule update --init --recursive 3rdparty/aiter
cd 3rdparty/aiter && git submodule update --init composable_kernel && cd ../..

# 必须禁用构建隔离，否则无法检测到 ROCm PyTorch
pip install . --no-build-isolation
```

---

## 2. 模型准备

模型下载到 `/shared_nfs/xiaofei/models/` 目录：

- **Qwen3-8B**：`/shared_nfs/xiaofei/Primus/output/qwen3_8b_safe_hf`（dense 模型，用于 smoke test）
- **Qwen3-30B-A3B**：`/shared_nfs/xiaofei/models/Qwen3-30B-A3B`（MoE 模型，128 experts，top-8）

---

## 3. 代码修改（必须）

以下修改是在 ROCm 上跑通 Megatron + sglang 的必要条件。

### 3.1 sglang torch_memory_saver adapter

**文件**: `sglang/python/sglang/srt/utils/torch_memory_saver_adapter.py`

`torch_memory_saver_numa` 的 API 与 `torch_memory_saver` 不同，会抛出 `AttributeError`，需要额外 catch：

```python
# 修改前
except ImportError as e:
# 修改后
except (ImportError, AttributeError) as e:
```

### 3.2 sglang rollout server 配置传递（verl 侧）

**文件**: `verl/workers/rollout/sglang_rollout/sglang_rollout.py`

原代码中 `_init_server_adapter` 没有将配置文件中的 `server.timeout` 等参数传递给 `AsyncHttpServerAdapter`，导致 MoE 模型 weight sync 使用默认超时（60s）不够用。

在 `_init_server_adapter` 方法中，创建 `AsyncHttpServerAdapter` 之前添加：

```python
host = f"[{server_address}]" if is_valid_ipv6_address(server_address) else server_address
server_kwargs = {}
if hasattr(self.config, "server") and self.config.server is not None:
    server_kwargs["timeout"] = getattr(self.config.server, "timeout", 60.0)
    server_kwargs["max_attempts"] = getattr(self.config.server, "max_attempts", 3)
    server_kwargs["retry_delay"] = getattr(self.config.server, "retry_delay", 2.0)
    server_kwargs["max_connections"] = getattr(self.config.server, "max_connections", 2000)
self._engine = AsyncHttpServerAdapter(
    model_path=self.model_config.local_path,
    host=host,
    port=server_port,
    launch_server=False,
    trust_remote_code=self.model_config.trust_remote_code,
    **server_kwargs,
)
```

### 3.3 Hydra rollout.yaml 添加 server 配置块

**文件**: `verl/trainer/config/rollout/rollout.yaml`

原始 yaml 缺少 `server:` 配置块，导致通过命令行传入 `rollout.server.timeout` 时 Hydra 报 `Key 'server' is not in struct` 错误。需要在 `prometheus:` 前添加：

```yaml
server:
  _target_: verl.workers.config.ServerConfig
  timeout: 60.0
  max_attempts: 3
  retry_delay: 2.0
  max_connections: 1000
  max_start_wait_time: 300.0
```

---

## 4. 训练脚本配置

### 4.1 关键环境变量

```bash
# ROCm / Ray GPU 分配
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1

# 禁用 CK fused attention（ROCm TE 不完全支持）
export NVTE_FUSED_ATTN_CK=0

# 【关键】禁用 aiter MoE weight shuffle
# aiter 的 process_weights_after_loading 会创建新的 Parameter 对象，
# 丢失 weight_loader 属性，导致 update_weights_from_tensor 失败
export SGLANG_USE_AITER=0
```

### 4.2 Qwen3-8B 配置要点（dense 模型 smoke test）

- 训练并行：TP=1, EP=1, PP=1（dense 模型不需要 EP）
- Rollout TP=1
- 无 CPU offload（8B 模型内存充足）
- `rollout.n=8`（GRPO 必须 > 1）
- `kl_loss_coef=0.04`, `kl_ctrl.kl_coef=0.04`
- 脚本见：`run_megatron_8b_smoke.sh`

### 4.3 Qwen3-30B-A3B 配置要点（MoE 模型）

- 训练并行：TP=1, EP=8, PP=1
- Rollout TP=4（sglang 推理用 4 卡 TP）
- 三重 CPU offload：`param_offload=True`, `optimizer_offload=True`, `grad_offload=True`
- Activation recompute：`recompute_method=uniform`, `recompute_granularity=full`
- 加大 server 超时：`timeout=600`, `max_attempts=10`
- `attention_backend='flash'`（fused 在 ROCm 上不可用）
- 脚本见：`run_megatron_30b.sh`

### 4.4 批量大小约束（容易出错）

verl 对 batch size 有严格的整除要求：

```
ppo_mini_batch_size >= DP_size
ppo_mini_batch_size % DP_size == 0
ppo_mini_batch_size <= train_batch_size
```

其中 `DP_size = world_size / (TP × PP)`。建议在脚本中加入 preflight check。

---

## 5. 问题排查记录

### 问题 1：torch_memory_saver 编译失败

- **现象**：`'assert' was not declared in this scope`
- **原因**：`csrc/torch_memory_saver_hip.cpp` 缺少 `#include <cassert>`
- **解决**：手动添加头文件后重新编译

### 问题 2：TransformerEngine 编译失败

- **现象**：`ValueError: ROCm is not supported by requested frameworks: ['pytorch']`
- **原因**：pip 隔离构建环境无法检测 ROCm PyTorch
- **解决**：`pip install . --no-build-isolation`

### 问题 3：TE aiter 子模块初始化卡住

- **现象**：`git submodule update --recursive` 卡在 `googletest`
- **原因**：嵌套子模块 `composable_kernel` 未正确初始化
- **解决**：只初始化必要子模块，手动初始化 `aiter/composable_kernel`

### 问题 4：sglang torch_memory_saver AttributeError

- **现象**：`module 'torch_memory_saver' has no attribute 'torch_memory_saver'`
- **原因**：sglang 只 catch `ImportError`，未处理 `torch_memory_saver_numa` 的 API 差异
- **解决**：修改 except 块增加 `AttributeError`

### 问题 5：Fused Attention 不可用

- **现象**：`ValueError: No dot product attention backend is available`
- **原因**：ROCm TE 的 fused attention 不完全支持
- **解决**：统一使用 `attention_backend='flash'`

### 问题 6：Hydra 配置覆盖语法

- **现象**：`+` 前缀添加已存在 key 报错
- **原因**：Hydra `+` 用于新增 key，已存在的需要 `++` 强制覆盖
- **解决**：ref 的 attention_backend 使用 `++` 前缀

### 问题 7：ZeroDivisionError（batch size 不整除）

- **现象**：`ZeroDivisionError: integer division or modulo by zero`
- **原因**：`ppo_mini_batch_size=2 < DP=8`，per-GPU batch size 为 0
- **解决**：设置 `ppo_mini_batch_size >= DP` 且能被 DP 整除

### 问题 8：Mode Collapse（GRPO n=1 + KL 系数过小）

- **现象**：训练 acc 从 90% 急剧崩塌至 20%，response length 从 430 坍缩到 14 tokens
- **根因**：
  1. `rollout.n=1` 使 GRPO advantage 计算退化（错误回答不被惩罚）
  2. `kl_coef=0.001` 太小，无法约束策略偏移
- **解决**：`rollout.n=8`, `kl_loss_coef=0.04`, `kl_ctrl.kl_coef=0.04`

### 问题 9：30B MoE weight sync 失败（最关键的 MoE 问题）

- **现象**：`RuntimeError: Failed to complete async request to update_weights_from_tensor after 3 attempts`
- **深层原因**：sglang 日志显示 `AttributeError: 'Parameter' object has no attribute 'weight_loader'`
- **根因分析**：
  - ROCm 上 `SGLANG_USE_AITER=1`（默认）时，sglang 的 `UnquantizedFusedMoEMethod.process_weights_after_loading` 调用 `shuffle_weight` 创建**新的** `torch.nn.Parameter` 对象
  - 新 Parameter 丢失了原始的 `weight_loader` 属性
  - `update_weights_from_tensor`（verl 权重同步）依赖 `weight_loader` 分发 expert 权重到 `FusedMoE` 层
  - 属性丢失导致权重同步崩溃，表现为超时错误
- **解决**：`export SGLANG_USE_AITER=0`
- **涉及文件**：`sglang/python/sglang/srt/layers/quantization/unquant.py`

### 问题 10：Hydra `server` key 不存在

- **现象**：`Key 'server' is not in struct`
- **原因**：`rollout.yaml` 没有定义 `server` 配置块
- **解决**：在 yaml 中手动添加 `server:` 块（见第 3.3 节）

### 问题 11：30B 训练 step 32 后 hang

- **现象**：训练正常跑了 32 步后，日志停止更新约 2.7 小时
- **诊断**：
  - 主进程 CPU 0%，状态 `futex_wait_queue_me`
  - sglang scheduler 4 个进程 CPU 73%~111%（busy-spinning，不处理请求）
  - sglang HTTP server 在 `ep_poll`（等待事件）
  - 无新错误日志
- **原因**：sglang scheduler 进入死循环或死锁（疑似 ROCm GPU 内存/通信问题）
- **状态**：多节点 2-node 训练中未复现

### 问题 12：Checkpoint 保存路径必须是 NFS 绝对路径（多节点）

- **现象**：checkpoint 保存后 HEAD 节点崩溃 (`Owner's node has crashed`)
- **原因**：`trainer.default_local_dir` 默认是相对路径，RayJob 工作目录是 `/sgl-workspace/`，checkpoint 写到了本地盘（无存储空间），磁盘写满导致节点崩溃
- **解决**：设置 `trainer.default_local_dir` 为 NFS 绝对路径，如 `/shared_nfs/xiaofei/verl/checkpoints/...`

### 问题 13：ROCm/HIP 7.0 fork 导致 checkpoint 保存 SIGSEGV（多节点）

- **现象**：checkpoint 保存时 `SIGSEGV received`，崩溃在 `crc32_16bytes()` → `torch.serialization._save`
- **调用链**：`write_preloaded_data_multiproc` → `Process.start()` (fork) → 子进程 `_write_item` → SIGSEGV
- **原因**：HIP 7.0 在 GPU 上下文初始化后 fork 子进程，子进程继承损坏的 HIP 内存映射
- **解决**：`verl/utils/megatron/dist_checkpointing.py` 中用 `threading.Thread` 替换 `multiprocessing.fork`，通过 `_patch_filesystem_writer_for_rocm()` 在每次 save 前自动应用
- **补丁内容**：
  - `_preload_no_pinned`：强制 `non_blocking=False`，避免 pinned memory 问题
  - `_write_threaded`：用线程并行写入替代 fork，保持 I/O 并行性能

### 问题 14：多节点 EP 场景必须强制使用 dist_checkpointing 格式

- **现象**：HF 格式保存时 hang 或崩溃
- **原因**：Megatron-Bridge 的 HF save 路径需要 `all_gather` 把分布在不同 EP rank 的 expert 权重聚合，跨节点的 `all_gather` 触发通信超时/崩溃
- **解决**：`megatron_workers.py` 中 force `use_dist_checkpointing=True, use_hf_checkpoint=False`
- **关于 dist_ckpt 格式**：基于 PyTorch Distributed Checkpoint，存储带 sharding 元信息的张量，支持并行度弹性（EP=8 保存 → EP=4 加载，自动重分片）

### 问题 15：__pycache__ 导致代码修改不生效（多节点）

- **现象**：修改了 `dist_checkpointing.py` 的补丁代码，但 RayJob 仍然使用旧逻辑（fork 方式）
- **原因**：RayJob 的 worker 节点从 NFS 加载了 `.pyc` 缓存文件，未重新编译 `.py`
- **解决**：训练脚本开头清理 `__pycache__`：
  ```bash
  find /shared_nfs/xiaofei/verl/verl/utils/megatron -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
  ```

### 问题 16：total_training_steps=-1 不兼容 Megatron lr_scheduler

- **现象**：`AssertionError: assert self.lr_decay_steps > 0`
- **原因**：`total_training_steps=-1` 表示无限训练，但 Megatron 的 `OptimizerParamScheduler` 需要具体的 decay 步数
- **解决**：不设置 `total_training_steps`，让 verl 根据 `total_epochs` 和数据量自动计算

### 问题 17：dist_ckpt → HuggingFace 格式转换

- **现象**：训练后的 checkpoint 无法直接用 sglang 推理
- **原因**：verl 没有内置的 dist_ckpt → HF 单机转换工具，`legacy_model_merger.py` 只支持旧格式
- **解决**：编写自定义转换脚本 `scripts/converter_mcore_to_hf.py`
  ```bash
  torchrun --nproc_per_node=8 scripts/converter_mcore_to_hf.py \
      --ckpt_dir .../global_step_N/actor \
      --hf_model_path /shared_nfs/xiaofei/models/Qwen3-30B-A3B \
      --output_dir /shared_nfs/xiaofei/models/Qwen3-30B-A3B-RL \
      --ep_size 8
  ```
  脚本工作流程：初始化 Megatron (EP=8) → CPU 创建模型 → 加载 dist_ckpt → 提取 HF 格式权重 → 各 rank 将 expert 权重写入 NFS 临时文件 → rank 0 合并保存为 safetensors

---

## 6. 训练结果

### 6.1 Qwen3-8B Megatron + sglang（已完成，成功）

| 指标 | 值 |
|------|------|
| 最终 GSM8K Acc | **83.5%** |
| 训练步数 | 467 步，完成 1 epoch |
| 平均吞吐量 | ~43 tok/s |
| 平均 step time | ~7s |
| GPU 显存 | 76.7 GB allocated / 78.6 GB reserved |
| 主要瓶颈 | weight sync ~2.6s（~50% step time） |

### 6.2 Qwen3-30B-A3B Megatron + sglang 单节点（32 步后 hang）

| 指标 | 值 |
|------|------|
| 运行步数 | 32/467（7%） |
| Step 20 Val Acc | 58.3% |
| Step 30 Val Acc | **64.5%**（有提升） |
| 平均 step time | ~68-70s |
| 平均吞吐量 | ~190-220 tok/s |
| GPU 显存 | 73.6 GB allocated / 84.3 GB reserved |
| CPU 内存 | ~869 GB |

step time 分解（30B 单节点）：

| 阶段 | 时间 | 占比 |
|------|------|------|
| update_actor（训练） | ~21s | 31% |
| update_weights（权重同步） | ~19s | 28% |
| gen（sglang rollout） | ~14s | 20% |
| old_log_prob | ~8s | 12% |
| ref | ~6s | 9% |

### 6.3 Qwen3-30B-A3B Megatron + sglang 2节点（已完成，成功）

| 指标 | 值 |
|------|------|
| 节点数 | 2 (16 GPU) |
| 并行策略 | 训练 TP=1,EP=8,PP=1,DP=2; Rollout TP=4,DP=2 |
| 训练步数 | 116 步，完成 1 epoch |
| 初始 GSM8K Acc | **43.8%** |
| 最终 GSM8K Score | **~80%** |
| 平均 step time | ~58s |
| 平均吞吐量 | ~430-500 tok/s |
| GPU 显存 | 94 GB allocated / 108 GB reserved (per GPU) |
| CPU 内存 | ~274 GB (per node) |
| 总训练时间 | 2h42m |
| Checkpoint 保存 | ✅ 线程写入，耗时 ~281s/次 |

step time 分解（30B 2节点）：

| 阶段 | 时间 | 占比 |
|------|------|------|
| gen（sglang rollout） | ~15s | 26% |
| update_weights（权重同步） | ~20s | 34% |
| update_actor（训练） | ~14s | 24% |
| old_log_prob | ~4s | 7% |
| ref | ~4s | 7% |

训练后的 HF 模型保存在 `/shared_nfs/xiaofei/models/Qwen3-30B-A3B-RL`，
由 `scripts/converter_mcore_to_hf.py` 从 dist_ckpt 转换而来。

**推理评测（GSM8K 前 100 题，sglang TP=4，temperature=0.6）**：

| 模型 | 正确率 |
|------|--------|
| Base (Qwen3-30B-A3B) | 87/100 (87%) |
| RL 训练后 | 96/100 (**96%**) |
| **提升** | **+9 个百分点** |

9 道分歧题 RL 全部答对，Base 全部答错。RL 训练不仅提升了推理能力，还改善了输出格式规范性（如 `75.00` → `75`）。

---

## 7. 已知问题与待解决项

1. **EP/TP 优化**：当前 EP=8/TP=1，Megatron 和 sglang 的模型切法不同（EP vs TP），update_weights 需要复杂的 expert 重映射（~20s/step，占 34%）。降低 EP 提高 TP 可能减少重分片开销
2. **Checkpoint 保存性能**：线程写入 ~281s/次，优化方向包括只保存 model 权重（跳过 optimizer）、减少保存频率
3. **async 训练**：verl 支持 sglang async rollout，可能提升整体 pipeline 效率
4. **ref 模型 offload**：关闭 `ref.megatron.param_offload` 可节省 ~3-4s/step
5. **dist_ckpt → HF 转换**：verl 缺少内置工具，当前使用自定义脚本 `scripts/converter_mcore_to_hf.py`，建议后续训练在 `save_contents` 中加入 `hf_model` 自动导出
