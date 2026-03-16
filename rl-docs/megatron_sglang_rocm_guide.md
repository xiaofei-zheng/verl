# verl Megatron + sglang RL 训练指南（ROCm MI355X）

环境：ROCm MI355X × 8 GPU，verl + Megatron-Core + sglang rollout，单节点 / 多节点

## Docker 镜像（推荐）

已提供预构建 Dockerfile，包含所有 Megatron 依赖（TransformerEngine、megatron-core、mbridge、torch_memory_saver）和 ROCm 适配补丁：

```bash
# 构建镜像（在 verl 仓库根目录执行）
docker build -f rl-docs/Dockerfile.megatron.rocm700.mi35x -t verl-megatron-rocm700:latest .

# 运行容器（挂载 NFS 和 GPU）
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    -v /shared_nfs:/shared_nfs \
    verl-megatron-rocm700:latest
```

基础镜像: `lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x`

Dockerfile 位于 `rl-docs/Dockerfile.megatron.rocm700.mi35x`，主要内容：
- 安装 TransformerEngine (ROCm)、megatron-core、mbridge、torch_memory_saver
- 安装 Ray 2.44.1
- 从 fork 仓库克隆 verl 并安装（包含所有 ROCm 适配代码修改）
- 设置 ROCm / NCCL / RCCL / Megatron 环境变量

如果使用 Docker 镜像，可直接跳到[训练脚本配置](#4-训练脚本配置)。

---

以下是手动配置步骤（不使用 Docker 镜像时参考）：

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

### 3.4 NCCL/RCCL 环境变量传播到 Ray workers

**文件**: `verl/trainer/constants_ppo.py`

多节点训练时，head 节点上设置的 NCCL/RCCL 优化环境变量（如 `NCCL_MIN_NCHANNELS`、`RCCL_MSCCL_ENABLE`、`HSA_NO_SCRATCH_RECLAIM` 等）不会自动传播到 Ray worker 进程，导致跨节点通信配置不一致，可能引发 RCCL 初始化 hang 或性能退化。

在 `get_ppo_ray_runtime_env()` 中添加自动环境变量传播逻辑：

```python
_NCCL_RCCL_PROPAGATE_PREFIXES = (
    "NCCL_", "RCCL_", "NCCL_NET_PLUGIN_PATH",
    "LD_LIBRARY_PATH", "HSA_NO_SCRATCH_RECLAIM",
    "GPU_MAX_HW_QUEUES", "TORCH_NCCL_HIGH_PRIORITY",
    "PYTORCH_HIP_ALLOC_CONF", "HIP_VISIBLE_DEVICES",
)

# 在 get_ppo_ray_runtime_env() 返回前添加：
for key, val in os.environ.items():
    if any(key.startswith(p) or key == p for p in _NCCL_RCCL_PROPAGATE_PREFIXES):
        runtime_env["env_vars"][key] = val
```

### 3.5 多节点 init_device_mesh 时序修复

**文件**: `verl/workers/megatron_workers.py`

**问题**：多节点训练大模型（如 Qwen3-235B）时，各 worker 从 NFS 加载模型的时间差异很大（可达 30 分钟以上）。原代码中 `init_device_mesh()` 在 `_build_rollout()` 中调用（即模型加载完成后），此时先完成加载的 worker 等待后完成的 worker 加入集体操作，但 Gloo 同步后端的 TCP rendezvous 超时（默认 1800s）会被耗尽，导致 `DistStoreError`。

**根因**：`init_device_mesh()` 是集体操作（所有 worker 必须同时调用 `new_group()`），但被放在了 worker 时间线已经分叉的位置。

**解决方案**：将 `init_device_mesh()` 从 `_build_rollout()` 移到 `__init__()` 中 `mpu.initialize_model_parallel()` 之后、模型加载之前。此时所有 worker 仍然同步，集体操作不会超时。

```python
# 在 __init__ 中，mpu.initialize_model_parallel() 之后添加：
if self._is_rollout:
    from torch.distributed.device_mesh import init_device_mesh

    infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
    infer_pp = self.config.rollout.pipeline_model_parallel_size
    infer_world_size = infer_tp * infer_pp
    world_size = torch.distributed.get_world_size()
    dp = world_size // infer_world_size
    self._rollout_device_mesh = init_device_mesh(
        get_device_name(),
        mesh_shape=(dp, infer_tp, infer_pp),
        mesh_dim_names=["dp", "infer_tp", "infer_pp"],
    )

# 在 _build_rollout() 中改为使用已创建的 device mesh：
rollout_device_mesh = self._rollout_device_mesh
self.rollout_device_mesh = rollout_device_mesh
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

### 问题 18：6 节点   B 训练 init_device_mesh DistStoreError（多节点）

- **现象**：6 节点 48 GPU 训练 Qwen3-235B 时，初始化阶段卡住 30 分钟后报 `torch.distributed.DistStoreError: wait timeout after 1800000ms`，错误发生在 `init_device_mesh` → `new_group()` 调用
- **根因**：
  1. 235B 模型从 NFS 加载需要很长时间（>30 分钟），各 worker 因 NFS I/O 竞争导致加载完成时间差异巨大
  2. 原代码中 `init_device_mesh()` 在 `_build_rollout()` 中调用（模型加载完成后），此时 worker 时间线已分叉
  3. `init_device_mesh()` 内部使用 Gloo 同步后端创建 process group，要求所有 worker 同时参与 TCP rendezvous
  4. 先完成加载的 worker 等待后完成的 worker 超过 Gloo 默认 1800s 超时，触发 `DistStoreError`
- **解决**：将 `init_device_mesh()` 从 `_build_rollout()` 移到 `__init__()` 中 `mpu.initialize_model_parallel()` 之后、模型加载之前（见 3.5 节）
- **注意**：单纯增加 `nccl_timeout` 无法解决此问题，因为超时发生在 Gloo 后端而非 NCCL

### 问题 19：NCCL/RCCL 环境变量未传播到 Ray workers（多节点）

- **现象**：多节点训练中部分 worker 的 RCCL 行为与 head 节点不一致，可能导致通信 hang 或性能退化
- **原因**：Ray worker 进程不继承 head 节点的 shell 环境变量，NCCL/RCCL 优化参数（如 `NCCL_MIN_NCHANNELS`、`RCCL_MSCCL_ENABLE`）丢失
- **解决**：修改 `verl/trainer/constants_ppo.py`，在 `get_ppo_ray_runtime_env()` 中自动传播以 `NCCL_`/`RCCL_` 等前缀开头的环境变量（见 3.4 节）

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

### 问题 20：8 节点 235B sglang init_memory_pool OOM

- **现象**：sglang 初始化 KV cache 时报 `RuntimeError: Not enough memory. mem_fraction_static=0.51`
- **根因**：verl 混合引擎中，actor 模型（EP=8, param_offload=False）占 ~72GB/GPU，sglang 模型（TP=8）占 ~59GB/GPU，合计 ~131GB。sglang 的 `gpu_memory_utilization` 控制其显存预算，但 sglang 会把 actor 的显存也算进 "model weights"，导致预算不够
- **尝试过的值**：gpu_memory_utilization=0.35（预算 85GB < 131GB → OOM）、0.5（122GB < 131GB → OOM）、0.6（147GB > 131GB 但仍不够，可能 NCCL 缓冲区等额外开销）
- **解决**：`gpu_memory_utilization=0.82`（预算 200GB）通过了 sglang init
- **关键对比**：30B 模型 actor+sglang 仅 22.5GB/GPU，`gpu_memory_utilization=0.6` 轻松够用

### 问题 21：8 节点 235B ref 计算 GPU OOM（free_cache_engine 未生效）

- **现象**：sglang init 通过后，`ref_compute_ref_log_prob` 报 `torch.OutOfMemoryError: 219.28 GiB allocated, 0 bytes free`
- **根因**：actor(72GB) + sglang model(59GB) + KV cache(~88GB) = 219GB，ref 模型 forward 需要额外 ~72GB → 超过 288GB
- **关键发现**：`free_cache_engine=True` 应该在 ref 计算前释放 KV cache，但**实际未生效**。可能是 verl 的 Megatron 混合引擎中 free_cache_engine 的生命周期管理有 bug
- **尝试 actor param_offload=True**：可以解决 GPU OOM，但导致 **CPU 内存 OOM**（每 worker offload ~144GB 到 CPU，8 workers/节点 = 1.15TB → 超出 cgroup 限制）
- **尝试 val_before_train=False**：跳过初始验证，但训练循环中 ref 计算同样 OOM
- **最终方案**：使用 PP=2 减半 actor 显存（见问题 22）

### 问题 22：PP=2 解决显存但引入 NCCL pipeline 超时

- **解决方案**：`TRAIN_PP=2`，actor 94 层分成 2 个 PP stage（47 层/stage），每 GPU actor 从 72GB 降到 ~36GB
- **效果**：PP=2 + gpu_memory_utilization=0.6 成功跑通 step 1
  - GPU 显存：143.2 GB allocated / 145.7 GB reserved（288GB 中只用一半）
  - Step time：143.8s，吞吐 52.6 tok/s，score 28.7%
- **新问题**：step 1 完成后，step 2 的 PP 跨节点通信 hang 30 分钟后超时
  - `PIPELINE_MODEL_PARALLEL_GROUP Rank 1: Watchdog caught collective operation timeout: ran for 1800007ms`
  - PP=2 要求跨节点 pipeline 通信（send/recv between PP stages），但 RCCL 跨节点 P2P 不稳定
- **状态**：待进一步调查 PP 跨节点通信问题，或寻找不用 PP 的替代方案

### 235B 8 节点显存分析

Qwen3-235B-A22B 模型参数（`moe_intermediate_size=1536`，非 `intermediate_size=12288`）：
- 非专家参数（attention + embed + lm_head）：~8B
- 专家参数（128 experts × 94 layers × 3 × 4096 × 1536）：~227.5B
- 总计：~235.5B

每 GPU 显存占用（288GB VRAM）：

| 组件 | PP=1,EP=8,TP=1 | PP=2,EP=8,TP=1 |
|------|----------------|----------------|
| Actor | 8B+28.4B=36.2B → **72GB** | ~18B → **36GB** |
| Sglang (TP=8) | 29.4B → **59GB** | **59GB** |
| 合计 | **131GB** (45%) | **95GB** (33%) |
| + Ref (forward) | +72GB → **203GB** | +36GB → **131GB** |
| + KV cache | OOM | 余量充足 |

### 问题 23：`torch_memory_saver` v0.0.5 API 不兼容 sglang（根因）

- **现象**：`enable_memory_saver=True` 时 sglang scheduler 报 `AttributeError: module 'torch_memory_saver' has no attribute 'torch_memory_saver'`
- **根因**：verl 的 Docker 镜像安装了 `torch_memory_saver_numa` v0.0.5（旧的 ROCm fork），缺少 sglang 要求的 tag-based API：
  - 无 `torch_memory_saver.torch_memory_saver` 模块级属性
  - `region()`/`pause()`/`resume()` 不接受 `tag` 参数
  - 无 `cuda_graph()`/`disable()` 方法
- **解决**：从主仓库 `fzyzcjy/torch_memory_saver` 源码构建 v0.0.9（PR #43 已包含 ROCm 支持，2025年8月合并）
  ```bash
  export HIPCC_COMPILE_FLAGS_APPEND="--amdgpu-target=gfx950 -D__HIP_PLATFORM_AMD__"
  export CFLAGS="-D__HIP_PLATFORM_AMD__"
  export CXXFLAGS="-D__HIP_PLATFORM_AMD__"
  pip install git+https://github.com/fzyzcjy/torch_memory_saver.git --no-deps --force-reinstall
  ```
- **验证**：v0.0.9 在 ROCm 7.0 / gfx950 上 tag-based pause/resume 正确释放物理显存（测试 535MB 释放+恢复，数据完整）
- **注意**：每个节点的容器需要独立安装（容器本地 `/opt/venv/`），训练脚本通过 Ray preflight 在所有节点自动安装
- **参考**：`fzyzcjy/torch_memory_saver` PR #43, PR #14; `verl-project/verl` PR #1464

### 问题 24：跨节点逐层通信 hang（TP>1, PP>1）

- **现象**：TP=2 或 PP=2 时，训练在 `compute_log_prob` 或 step 2 hang（GPU 100% 但 Memory bandwidth 0%，30分钟后 NCCL 超时）
- **根因**：TP 和 PP 要求**逐层跨节点通信**（TP all-reduce 每层 2 次 × 94 层 = 188 次/forward，PP send/recv 每层）。这种高频小块跨节点通信模式在 RCCL + AINIC 网络上 hang
- **对比**：30B 用 TP=1 EP=8 DP=8 在同一集群正常运行，因为只有 DP gradient sync（一次性大块跨节点通信）能正常工作
- **结论**：当前集群可行配置限制为 **TP=1, PP=1**（节点内模型并行），只有 DP 做跨节点通信
- **待查**：RCCL/AINIC 的逐层跨节点 P2P 通信问题

### 问题 25：`gpu_memory_utilization` 与 `enable_memory_saver` 的 tradeoff

- **矛盾**：sglang init 需要 `gpu_memory_utilization≥0.82` 才能通过（actor 73GB + sglang 59GB + NCCL overhead ≈ 170GB+），但 0.82 分配了 ~70GB KV cache，使得 ref 阶段 VRAM 接近 99%
- **memory_saver 部分有效**：v0.0.9 的 `pause(tag="kv_cache")` 确实释放了物理显存（观察到 VRAM 从 87% 降到 41%），但 ref forward 时 actor(73GB) + ref 参数从 CPU 加载(73GB) + 残留分配 → 重新填满到 99%
- **expandable_segments 无关**：去掉 `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` 未改善问题
- **状态**：需要进一步调查 ref forward 阶段为什么 VRAM 重新填满到 99%（memory_saver 释放了 KV cache 但可能未释放 sglang model weights，或 ref param_offload 的 CPU→GPU 加载策略有问题）

### 问题 26：`torch_memory_saver` v0.0.5 → v0.0.9 升级（ROCm 适配）

- **根因**：verl Docker 镜像中的 `torch_memory_saver_numa` v0.0.5 缺少 sglang 要求的 tag-based API
- **解决**：从 `fzyzcjy/torch_memory_saver` 主仓库源码构建 v0.0.9（PR #43 包含 ROCm/HIP 支持）
  ```bash
  export HIPCC_COMPILE_FLAGS_APPEND="--amdgpu-target=gfx950 -D__HIP_PLATFORM_AMD__"
  pip install git+https://github.com/fzyzcjy/torch_memory_saver.git --no-deps --force-reinstall
  ```
- **验证**：v0.0.9 在 ROCm 7.0 / gfx950 上 tag-based pause/resume 正确释放物理显存（535MB 测试通过）
- **部署**：训练脚本通过 Ray preflight 在所有节点自动安装（容器重启后需重新安装）

### 问题 27：ref_compute_ref_log_prob 阶段 VRAM 重新填满（核心阻塞问题）

- **现象**：`enable_memory_saver=True` + v0.0.9 下，generation 完成后 `sleep_replicas` 成功将 VRAM 从 87% 降到 41%，`compute_old_log_prob` 正常完成，但 `ref_compute_ref_log_prob` 阶段 VRAM 涨回 95-99% 并 crash（SYSTEM_ERROR / GPU hang）
- **15 次运行对比得出的结论**：
  - `gpu_memory_utilization < 0.82` → sglang init OOM（预算不够 actor 73GB + sglang 59GB + overhead）
  - `gpu_memory_utilization ≥ 0.82` → sglang init 通过，但分配 ~70GB KV cache → ref 阶段 VRAM 满
  - `enable_memory_saver` 释放了 KV cache 物理内存（VRAM 降到 41%），但 ref forward 时 VRAM 重新涨到 99%
- **疑似根因**：`enable_memory_saver` 可能只释放了 KV cache（~70GB）但**未释放 sglang model weights（59GB）**，导致 ref 阶段实际显存为 actor(73GB) + sglang_weights(59GB) + ref(73GB) + overhead ≈ 240GB → 接近 288GB
- **actor param_offload=True 尝试**：可以解决 GPU 显存问题（actor 不占 GPU），但需要 CPU 内存 ≥ 2.8TB/节点（8 workers × ~250GB RSS）
- **官方 235B 示例对比**：官方用 vLLM（不是 sglang） + PP=8 + TP=4 + param_offload=True，完全不同的架构

### 235B 8 节点 15 次运行总结

| 阶段 | 状态 | 有效配置 |
|------|------|---------|
| sglang init | 已解决 | `gpu_memory_utilization=0.82` |
| generation | 已解决 | TP=1 EP=8（节点内），GPU 100% 正常推理 |
| sleep_replicas 释放显存 | 已解决 | `enable_memory_saver=True` + v0.0.9，VRAM 87%→41% |
| compute_old_log_prob | 已解决 | 正常完成 |
| **ref_compute_ref_log_prob** | **未解决** | VRAM 涨回 99% 后 crash |
| 跨节点 TP>1/PP>1 | 不可行 | RCCL 逐层跨节点通信 hang |

---

### 235B 8 节点集群重启后排查（2026-03-13 ~ 03-14）

排除坏节点 10.158.172.63 后集群重启，新一轮 debug（~6 次尝试）：

#### 问题 1：update_weights 阶段 Actor 进程被杀

- **现象**：进入 `trainer.fit()` → `checkpoint_manager.update_weights()` 后，Ray actor 死亡
- **错误类型**：前两次 `ActorDiedError`（worker 进程 SIGKILL），第三次 `ActorUnavailableError`（RPC Socket closed）
- **每次死亡节点不同**：10.158.169.78 → 10.158.169.78 → 随机，说明不是特定坏节点
- **根因分析**：`update_weights` 调用链内存极高
  - `per_tensor_generator` 中 EP=8 的 `all_gather` 把 128 个 expert 权重在每个 rank 上重组
  - `get_named_tensor_buckets` 对每个 tensor 做 `clone()`，每 bucket 2GB 额外拷贝
  - actor(~58GB/rank) + sglang weights + clone buffer 同时在 GPU → OOM
- **尝试的优化**：
  - `update_weights_bucket_megabytes` 从 2048 → 512 → 256
  - `gpu_memory_utilization` 从 0.82 → 0.6
  - `enable_memory_saver=True`（sglang 侧 TMS 释放权重）
  - `use_mbridge=True`（绕过 per_tensor_generator 的 EP all_gather）
  - 增加 Ray 超时（health check 300s, gRPC 900s）

#### 问题 2：torch_memory_saver v0.0.5 vs v0.0.9 不兼容

- **现象**：`AttributeError: module 'torch_memory_saver' has no attribute 'torch_memory_saver'`
- **原因**：集群重启后新 pod 自带 v0.0.5（镜像默认），sglang 需要 v0.0.9
- **preflight 不可靠**：pip install from git 在部分节点超时（>300s），安装不一致
- **解决方案**：**禁用 `enable_memory_saver=False`**，完全避免 TMS 依赖
- **代价**：sglang 不能在 sleep 时释放 model weights，但 `free_cache_engine=True` 仍能释放 KV cache

#### 问题 3（最新）：update_actor OOM — 真正的瓶颈

- **现象**：`update_weights` 成功通过，但 `_update_actor` → `load_megatron_model_to_gpu` 时 OOM
- **错误**：`Tried to allocate 105.75 GiB. GPU has 287.98 GiB total, 83.05 GiB free`
- **根因**：`enable_memory_saver=False` 时，sglang model weights（~160GB/8GPU）没被释放；actor 要加载 params+grad 需要 ~106GB，只有 83GB 空闲
- **显存占用分析**（288GB/GPU）：

| 组件 | 占用 | 说明 |
|------|------|------|
| sglang weights（未释放） | ~160 GB | 无 memory_saver，weights 常驻 |
| KV cache（已释放） | 0 GB | free_cache_engine=True |
| PyTorch 分配 | ~45 GB | actor ref/其他 |
| **剩余** | ~83 GB | 不够 actor 105GB |

- **解决方向**：
  1. **修复 TMS 安装**：用 NFS 上预编译 wheel 替代 git install，保证 `enable_memory_saver=True` 可用
  2. **降低 actor grad 占用**：只加载 params 不加载 grad（`load_grad=False`），但 update_actor 需要 grad
  3. **降低 gpu_memory_utilization 到 0.4-0.5**：让 sglang weights 占更少
  4. **切换 vLLM**：官方 235B 方案用 vLLM，其 weight sync 不需要 TMS

### 当前配置快照（最新尝试）

```bash
# 并行
NNODES=8, TP=1, EP=8, PP=1, DP=8, ROLLOUT_TP=8

# 显存
param_offload=True, optimizer_offload=True, grad_offload=True
gpu_memory_utilization=0.6, free_cache_engine=True
enable_memory_saver=False  # 禁用以避免 TMS 依赖
update_weights_bucket_megabytes=256

# 优化
use_mbridge=True  # 高效权重转换，绕过 per_tensor_generator EP all_gather
```

### 后续修复（2026-03-14 凌晨）

#### TMS NFS Wheel 方案（已验证可行）
- 在 head 节点 `pip wheel git+https://github.com/fzyzcjy/torch_memory_saver.git --no-deps -w /shared_nfs/xiaofei/wheels/`
- Preflight 从 NFS 安装：16 秒装完 8 节点，100% 成功率
- SGLang 启动成功，`enable_memory_saver=True` 生效
- 显存释放后 sglang CUDA graph capture 可用内存 136.76 GB

#### vLLM 切换尝试（失败）
- vLLM v0.9.2 (rocm700 build) 存在多个 ROCm 不兼容问题：
  - `is_sleep_mode_available()` 返回 False（已 patch）
  - `cutlass_scaled_mm_supports_fp8` 不存在（CUTLASS 是 NVIDIA 的，`is_cuda()` 在 ROCm 上返回 True 触发）
  - `vllm.vllm_flash_attn.layers` 模块不存在
- **结论**：当前环境的 vLLM 在 ROCm 上不可用作 rollout，需要 sglang

#### 当前瓶颈：Ray RPC 网络不稳定（ActorUnavailableError）
- `enable_memory_saver=True` + `mbridge=True` + NFS wheel → SGLang 初始化成功，进入训练循环
- `update_weights` 跑了 ~20 分钟（传输 235B 权重），然后 `ActorUnavailableError: Socket closed`
- **不是内存问题，不是代码问题——是集群网络稳定性问题**
- 235B 的 update_weights 需要通过 Ray RPC 传输大量数据（EP=8 all_gather + bucket transfer），长时间的阻塞操作容易被网络抖动打断

### 下一步方案

1. **集群网络排查**：联系 infra 团队检查 8 节点间的 RDMA/IB 网络稳定性
2. **NCCL checkpoint engine**：用 `backend=nccl` 替代 `naive`，通过 NCCL 传输权重（比 Ray RPC 更稳定）
3. **减少 update_weights 时间**：增大 `update_weights_bucket_megabytes`（当前 256，可以试 1024），减少 RPC 往返次数
4. **重试机制**：在 verl 的 `update_weights` 调用处添加重试逻辑

---

### 235B 4 节点训练成功跑通（2026-03-14）

#### 方案概述

放弃 8 节点 TP=1 PP=1 EP=8 DP=8 方案（Ray RPC 在 update_weights 跨 DP 组同步时超时），改用官方推荐的 4 节点 (32 GPU) 配置：

| 参数 | 8 节点（失败） | 4 节点（成功） |
|------|---------------|---------------|
| TP | 1 | **4** |
| PP | 1 | **8** |
| EP | 8 | **4** |
| DP | 8 | **1** |
| Rollout | sglang TP=8 | sglang TP=8 |
| OFFLOAD_FRACTION | 1 | 1 |

关键改进：DP=1 意味着 `update_weights` 不需要跨 DP 组同步（8 节点 DP=8 需要 64 个 worker 协调 → Ray RPC 超时），只需要 1 个模型副本内部的 rollout↔training 权重同步。

#### 并行策略分析

```
32 GPUs = TP(4) × PP(8) × DP(1)
EP=4 fits within TP×DP = 4×1 = 4 GPUs per PP stage
每个 PP stage: 94/8 ≈ 12 层, 分布在 4 个 GPU (TP=4, EP=4)
```

PP=8 需要跨节点 pipeline 通信（send/recv），但比 TP 的 all-reduce 更轻量。之前问题 24 记录的 "TP>1/PP>1 跨节点通信 hang" 在 4 节点配置下**未复现**。

#### 基于官方脚本

基于 `examples/grpo_trainer/run_qwen3-235b_megatron_96gb.sh`，适配 ROCm：

- **Rollout**: vLLM → **sglang**（vLLM 在 ROCm 上不可用，见问题 vLLM 切换尝试）
- **去掉 NVIDIA-only fused kernels**: `moe_enable_deepep`, `moe_permute_fusion`, `gradient_accumulation_fusion`, `apply_rope_fusion`, `moe_token_dispatcher_type=flex`
- **保留 ROCm 环境变量**: RCCL 网络配置、TMS NFS wheel 安装
- **保留**: `enable_memory_saver=True`, `use_mbridge=True`, `attention_backend=flash`
- **保留官方 batch config**: `train_batch_size=32`, `mini_batch=16`, `n=8`, `lr=1e-6`

脚本已删除（被 `rl-scripts/run_megatron_235b_8node_4gpu.sh` 取代）

#### 训练结果（前 9 步）

| 指标 | 值 |
|------|------|
| 节点数 | 4 (32 GPU, MI355X 288GB) |
| 训练步数 | 9/233（step 10 checkpoint 保存时崩溃） |
| 初始 GSM8K Score | ~43% |
| Step 9 Score | **63.7%** |
| 平均 step time | ~350s (~5.8 min) |
| 平均吞吐量 | ~37-43 tok/s |
| GPU 显存 | 51.5 GB allocated / 72.8 GB reserved (288GB 中) |
| CPU 内存 | ~1545-1598 GB (4 节点合计) |
| update_weights 时间 | ~91s/step（**稳定，无超时**） |
| update_actor 时间 | ~34s/step |
| gen (sglang) 时间 | ~197s/step |
| 预计总训练时间 | ~22.5 小时 |

Step time 分解（235B 4 节点，以 step 2 为例）：

| 阶段 | 时间 | 占比 |
|------|------|------|
| gen（sglang rollout） | ~198s | 58% |
| update_weights（权重同步） | ~91s | 27% |
| update_actor（训练） | ~34s | 10% |
| old_log_prob | ~11s | 3% |
| ref | ~8s | 2% |

#### 已知问题：Checkpoint 保存崩溃

训练循环稳定运行，但 `save_freq=10` 在 step 10 触发 `_save_checkpoint()` 时崩溃：

```
ray.exceptions.ActorUnavailableError: Socket closed
调用链: trainer.fit() → _save_checkpoint() → actor_rollout_wg.save_checkpoint() → ray.get(output)
```

根因分析：
1. **不是 SIGSEGV**（ROCm fork→thread 补丁已生效）
2. **不是 HF 格式 all_gather hang**（dist_checkpointing 格式已强制启用）
3. **是 Ray RPC 超时**：235B 模型 32 个 worker 同时向 NFS 写入 dist_ckpt（30B 用了 ~281s/次，235B 预计更久），worker 在写入期间不响应 Ray keepalive → Socket closed

**深层根因**（GCS 日志分析）：

不是 Ray RPC 超时，而是**远程节点 10.158.171.199 的 raylet 进程完全崩溃**：

```
11:41:57 GCS: Connection is broken. node_id=6a577...
11:41:58 GCS: Health check FAILED for node 6a577...
         ipv4:10.158.171.199:33151: Connection refused
11:42:07 GCS: Destroying actor a096d5ca... (on crashed node)
```

同时 head 节点 raylet 持续报 `memory_monitor.cc: Got negative used memory for cgroup -1`（cgroup 内存监控异常）。推测崩溃原因：

1. 训练期间 CPU 内存 ~400 GB/节点（4 节点合计 1598 GB）
2. `save_checkpoint` → `load_megatron_model_to_gpu(load_grad=True)` + `save_dist_checkpointing` 的 NFS 写入缓冲 → CPU 内存峰值超过 cgroup 限制
3. Linux OOM Killer 终止了该节点的 raylet 进程
4. **与 30B 的 SIGSEGV/fork 问题是完全不同的根因**（30B CPU 内存仅 ~274 GB/节点，远低于限制）

另外 `save_dist_checkpointing` 流程中有两处 `all_gather_object` 跨 32 rank 集合通信（`validate_access_integrity`），可能加剧内存压力。

潜在修复方向：
1. **减少 save 时 CPU 峰值**：`load_grad=False`（只保存 params 不保存 grad）
2. **跳过 validation**：`validate_access_integrity=False` 减少 `all_gather_object` 内存开销
3. **增大 cgroup memory limit**：联系 infra 调整容器内存限制
4. **异步 checkpoint**：`async_save=True` 避免阻塞

临时解决方案：`save_freq=-1` 跳过 checkpoint 保存，让训练先跑通（预计 ~22.5 小时）。

### 问题 28：save_checkpoint 时 ROCm HIP RSS 膨胀导致 cgroup OOM（根因与解决）

- **根因深入分析**：
  1. ROCm HIP 将 GPU 显存映射到进程虚拟地址空间，导致每个 worker 的 RSS 包含 GPU 显存占用（每 worker ~350GB）
  2. 原代码 `save_checkpoint` 调用 `load_megatron_model_to_gpu(self.actor_module)` 将 offload 到 CPU 的参数重新加载到 GPU → 新增 GPU 显存 → RSS 膨胀
  3. 4 节点 × 8 GPU/节点 = 8 workers/节点，RSS 总量 ~2800GB ≈ cgroup 限制，任何额外分配都触发 OOM
- **解决方案 1：CPU-based checkpoint save**（代码修改）
  - 新增 `_swap_params_to_cpu_for_save()` 方法，直接使用 CPU offload 副本生成 state_dict，完全避免 GPU 分配
  - 修改 `save_checkpoint()` 调用流程：跳过 `load_megatron_model_to_gpu()`，改用 CPU swap
  - **文件**: `verl/workers/megatron_workers.py`
- **解决方案 2：减少每节点 worker 数量**
  - 从 4 节点 × 8 GPU 改为 **8 节点 × 4 GPU**，总 GPU 数不变（32 张）
  - 每节点 4 workers × 350GB = 1400GB，远低于 cgroup 限制，留出 1400GB 余量
  - 代价：占用更多节点，GPU 4-7 空闲
- **解决方案 3：跳过 ROCm 上的 sharding validation**
  - `save_dist_checkpointing` 中的 `validate_sharding_integrity` 调用 `all_gather_object` 收集所有 rank 的 sharding 信息，内存开销大
  - ROCm 上禁用此验证：`validate_sharding_integrity = not is_rocm`
  - **文件**: `verl/utils/megatron/dist_checkpointing.py`
- **最终配置**：方案 1 + 方案 2 + 方案 3 组合使用，checkpoint 保存稳定通过
- **脚本见**: `rl-scripts/run_megatron_235b_8node_4gpu.sh`（包含 inline patch，打入镜像后可去掉）

### 235B 4 节点 train_only 完整训练（2026-03-14 ~ 03-15）

使用 `save_freq=-1` 跳过 checkpoint 保存，验证训练循环本身可以跑通：

| 指标 | 值 |
|------|------|
| 脚本 | `rl-scripts/run_megatron_235b_4node_train_only.sh` |
| 节点数 | 4 (32 GPU, MI355X 288GB) |
| 并行策略 | TP=4, PP=8, EP=4, DP=1 |
| 训练步数 | **233/233（100% 完成）** |
| 总训练时间 | **17 小时 42 分钟** |
| 平均 step time | ~273s (~4.5 min) |
| 最终 score/mean | ~0.95（95%） |
| Checkpoint | 无（save_freq=-1） |

**结论**：训练循环完全稳定，233 步无任何 crash。问题仅在 checkpoint 保存阶段。

### 235B 8 节点 × 4GPU 完整训练（最终方案，2026-03-15 ~ 03-16）

应用 CPU-save patch + 8 节点 4GPU 配置，训练 + checkpoint 保存全部跑通：

| 指标 | 值 |
|------|------|
| 脚本 | `rl-scripts/run_megatron_235b_8node_4gpu.sh` |
| 节点数 | 8 (每节点 4 GPU, 共 32 GPU) |
| 并行策略 | TP=4, PP=8, EP=4, DP=1; Rollout TP=8 |
| 训练步数 | **233/233（100% 完成）** |
| 总训练时间 | **16 小时 50 分钟** |
| 平均 step time | ~280-330s (~5 min) |
| 最终 score/mean | ~0.95（95%） |
| Checkpoint 保存 | **全部成功**（save_freq=50） |

Checkpoint 保存记录：

| Checkpoint | 耗时 | 状态 |
|------------|------|------|
| global_step_50 | ~35 min | OK |
| global_step_100 | ~35 min | OK |
| global_step_150 | ~35 min | OK |
| global_step_200 | ~35 min | OK |
| global_step_233（最终） | ~40 min | OK |

Checkpoint 大小分析：

| 内容 | 精度 | 大小 |
|------|------|------|
| 模型权重 | bf16 | ~470 GB |
| Adam optimizer momentum | fp32 | ~940 GB |
| Adam optimizer variance | fp32 | ~940 GB |
| fp32 master weights | fp32 | ~940 GB |
| **单个 checkpoint 总计** | | **~3 TB** |

每个 checkpoint 包含 66 个 `.distcp` 文件（32 rank × 2 bucket），每文件 46-49 GB。`max_ckpt_to_keep` 自动清理旧 checkpoint，最终保留 global_step_200 和 global_step_233。

**结论**：这是 Qwen3-235B 在 ROCm MI355X 上的首次完整 RL 训练 + checkpoint 保存成功。

---

## 7. 已知问题与待解决项

### 已解决

1. ~~**235B checkpoint 保存 OOM**~~：通过 CPU-save patch + 8 节点 4GPU 配置解决（见问题 28）
2. ~~**TMS 安装可靠性**~~：NFS 预编译 wheel 方案已验证可行（16 秒装完 8 节点）
3. ~~**235B 完整训练**~~：233 步 + 5 个 checkpoint 全部成功（见 6.5 节）

### 待优化

1. **Checkpoint 大小**：每个 checkpoint ~3 TB（含 optimizer states），仅模型权重 ~470 GB。如只需推理，可配置只保存模型权重跳过 optimizer（节省 ~85% 空间）
2. **Checkpoint 保存耗时**：单次 ~35-40 分钟（235B 写入 NFS），占训练时间 ~4%
3. **每节点只用 4 GPU**：当前方案 GPU 4-7 空闲，未来修复 ROCm HIP RSS 映射问题或调高 cgroup 限制后可改回 8 GPU/节点
4. **EP/TP 优化**：当前 TP=4 EP=4，update_weights 需要 expert 重映射。进一步优化 EP/TP 比例可能减少重分片开销
5. **async 训练**：verl 支持 sglang async rollout，可能提升整体 pipeline 效率
6. **dist_ckpt → HF 转换**：verl 缺少内置工具，当前使用自定义脚本 `scripts/converter_mcore_to_hf.py`
7. **代码修改需打入镜像**：3 个 ROCm 适配修改（`async_sglang_server.py`、`megatron_workers.py`、`dist_checkpointing.py`）目前由训练脚本 inline patch，打新镜像时应直接包含这些修改
