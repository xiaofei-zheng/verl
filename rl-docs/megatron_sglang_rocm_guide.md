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
- **状态**：待进一步排查，需要 kill 后重跑观察复现性

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

### 6.2 Qwen3-30B-A3B Megatron + sglang（32 步后 hang）

| 指标 | 值 |
|------|------|
| 运行步数 | 32/467（7%） |
| Step 20 Val Acc | 58.3% |
| Step 30 Val Acc | **64.5%**（有提升） |
| 平均 step time | ~68-70s |
| 平均吞吐量 | ~190-220 tok/s |
| GPU 显存 | 73.6 GB allocated / 84.3 GB reserved |
| CPU 内存 | ~869 GB |

step time 分解（30B）：

| 阶段 | 时间 | 占比 |
|------|------|------|
| update_actor（训练） | ~21s | 31% |
| update_weights（权重同步） | ~19s | 28% |
| gen（sglang rollout） | ~14s | 20% |
| old_log_prob | ~8s | 12% |
| ref | ~6s | 9% |

---

## 7. 已知问题与待解决项

1. **30B 训练 hang 问题**：step 32 后 sglang scheduler 死锁，需要排查是否可复现以及具体触发条件
2. **EP/TP 优化**：当前 EP=8/TP=1，可以尝试降低 EP 提高 TP 来减少 expert 重分片开销
3. **外部开发者 patch**：参考 verl MoE Megatron 适配代码，可能有更好的权重同步方案
4. **ref 模型 offload 优化**：当前 `ref.megatron.param_offload=True`，关闭后预计可节省 3-4s/step（~5%）
5. **async 训练**：verl 支持 sglang async rollout，可能提升整体 pipeline 效率
