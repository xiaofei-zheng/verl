# verl GRPO 训练 ROCm 环境配置指南

基于 sglang 初始镜像，配置 verl + FSDP + SGLang rollout 在 MI355X 上运行 GRPO 训练的完整步骤。

## 前提条件

- 镜像：sglang ROCm 镜像（已包含 sglang、torch、ROCm 等）
- GPU：AMD MI355X（8 卡）
- 模型：已在 `/shared_nfs` 上准备好的 HuggingFace 格式模型
- verl 源码：`/shared_nfs/xiaofei/verl`

## 第一步：安装 verl

sglang 初始镜像不包含 verl 及其依赖（如 `tensordict`、`hydra-core` 等），需要从源码安装：

```bash
cd /shared_nfs/xiaofei/verl
pip install -e .
```

关键依赖会自动安装，包括：`tensordict`、`hydra-core`、`omegaconf`、`torchdata`、`wandb` 等。

## 第二步：修改 verl 源码适配 ROCm

需要修改 1 个文件：`verl/workers/rollout/sglang_rollout/async_sglang_server.py`

### 修改 1：添加 ROCm 检测函数

在文件顶部（import 区域之后）添加：

```python
def _is_rocm() -> bool:
    return hasattr(torch.version, 'hip') and torch.version.hip is not None
```

### 修改 2：禁用 memory_saver

将 `enable_memory_saver` 从 `True` 改为 `False`：

```python
# 原始代码：
"enable_memory_saver": True,

# 修改为：
"enable_memory_saver": False,
```

原因：`torch_memory_saver` 依赖 NVIDIA 的 `libcuda.so.1`，在 ROCm 环境中不存在。

### 修改 3：attention backend 适配

将 `mm_attention_backend` 和 `attention_backend` 改为 ROCm 条件判断：

```python
# 原始代码：
"mm_attention_backend": "fa3",
"attention_backend": attention_backend if attention_backend is not None else "fa3",

# 修改为：
"mm_attention_backend": "fa3" if not _is_rocm() else "aiter",
"attention_backend": attention_backend if attention_backend is not None else ("aiter" if _is_rocm() else "fa3"),
```

原因：FA3（Flash Attention 3）是 NVIDIA Hopper 专属，ROCm 使用 `aiter` 后端。

## 第三步：清理 aiter JIT 锁文件（如果需要）

每次 pod 重启或镜像恢复后，可能残留 aiter JIT 编译锁文件，导致 sglang 启动 hang：

```bash
find /sgl-workspace/aiter/aiter/jit/build -name "lock*" -exec rm -f {} \; 2>/dev/null
```

首次启动时 aiter 会 JIT 编译所有 kernel（rmsnorm、custom_all_reduce 等），这个过程约需 **10-15 分钟**，是一次性开销，编译产物会缓存在 `/sgl-workspace/aiter/aiter/jit/` 下。

## 第四步：恢复 aiter JIT 编译缓存（跳过 10-15 分钟编译）

首次编译完成后，编译产物已备份到 NFS。新 pod 启动后直接复制回来即可跳过编译：

```bash
cp /shared_nfs/xiaofei/aiter_jit_cache/*.so /sgl-workspace/aiter/aiter/jit/
cp -r /shared_nfs/xiaofei/aiter_jit_cache/build /sgl-workspace/aiter/aiter/jit/
```

如果需要打进镜像，把 `/sgl-workspace/aiter/aiter/jit/` 整个目录 COPY 到镜像中即可（约 546M）。

验证缓存是否生效：

```bash
ls /sgl-workspace/aiter/aiter/jit/module_rmsnorm.so && echo "JIT cache OK"
```

如果没有缓存也没关系，首次启动训练时会自动编译（约 10-15 分钟），编译期间日志会显示 `waiting for baton release at lock_module_rmsnorm`。

## 第五步：设置环境变量

```bash
# GPU 设备（ROCm 必须）
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES

# Ray 防止覆盖 GPU 设备变量
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1

# 禁用 FA3（sglang 启动参数）
export SGLANG_DISABLE_FA3=1

# NCCL/RCCL 优化参数
export NCCL_MIN_NCHANNELS=112
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
export NCCL_CROSS_NIC=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0

# 其他
export TOKENIZERS_PARALLELISM=false
export HSA_NO_SCRATCH_RECLAIM=1
```

## 第六步：启动训练

```bash
cd /shared_nfs/xiaofei/verl

MODEL_PATH=/shared_nfs/xiaofei/Primus/output/qwen3_8b_safe_hf
GPUS_PER_NODE=8

PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/shared_nfs/xiaofei/verl/data/gsm8k/train.parquet \
    data.val_files=/shared_nfs/xiaofei/verl/data/gsm8k/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=verl_grpo_qwen3_8b \
    trainer.experiment_name=qwen3_8b_grpo_gsm8k \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.default_local_dir=/shared_nfs/xiaofei/verl_checkpoints \
    trainer.test_freq=5 \
    trainer.total_epochs=2
```

## 关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `actor.strategy` | `fsdp2` | 使用 FSDP2（PyTorch >= 2.4，ROCm 兼容） |
| `rollout.name` | `sglang` | 使用 sglang 做推理（非 vLLM） |
| `rollout.tensor_model_parallel_size` | `8` | 所有 GPU 归一个 sglang server，避免多 server 内存竞争 |
| `actor.fsdp_config.param_offload` | `True` | 参数卸载到 CPU，为 sglang 腾出 GPU 内存 |
| `actor.fsdp_config.optimizer_offload` | `True` | 优化器状态卸载到 CPU |
| `actor.use_torch_compile` | `True` | 开启 `torch.compile` 加速训练（FSDP2 支持） |
| `rollout.gpu_memory_utilization` | `0.4` | sglang KV cache 使用 40% GPU 内存 |

## 训练后合并 checkpoint

训练产生的 FSDP 分片 checkpoint 需要合并为 HuggingFace 格式才能推理：

```bash
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /shared_nfs/xiaofei/verl_checkpoints/global_step_XXX/actor \
    --target_dir /shared_nfs/xiaofei/verl_checkpoints/merged_model
```

## 常见问题

### 1. `ModuleNotFoundError: No module named 'tensordict'`

原因：pod 重启后 pip 包丢失。解决：`pip install -e /shared_nfs/xiaofei/verl`

### 2. `libcuda.so.1: cannot open shared object file`

原因：`enable_memory_saver=True` 触发加载 NVIDIA 库。解决：确认第二步修改 2 已生效。

### 3. `ImportError: Can not import FA3 in sgl_kernel`

原因：FA3 是 NVIDIA 专属。解决：确认第二步修改 3 已生效，且 `export SGLANG_DISABLE_FA3=1`。

### 4. sglang 启动时 hang 在 `waiting for baton release at lock_module_rmsnorm`

两种情况：
- 首次编译：正常等待 10-15 分钟，`ps aux | grep hipcc` 能看到编译进程。
- 残留锁：执行第三步清理锁文件，然后重启训练。

### 5. `RuntimeError: Not enough memory`（sglang KV cache）

原因：多个 sglang server 竞争 GPU 内存。解决：使用 `tensor_model_parallel_size=8`（所有 GPU 归一个 server）+ FSDP offload。

### 6. `ValueError: Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set`

原因：verl 内部逻辑冲突。解决：只设置 `HIP_VISIBLE_DEVICES`，`unset ROCR_VISIBLE_DEVICES` 和 `unset CUDA_VISIBLE_DEVICES`。

## 快速一键脚本

将以下内容保存为 `run_verl_grpo.sh`，可在任何恢复后的 sglang 初始镜像 pod 上直接运行：

```bash
#!/bin/bash
set -e

# 1. 安装 verl
pip install -e /shared_nfs/xiaofei/verl

# 2. 恢复 aiter JIT 缓存（跳过 10-15 分钟编译）
if [ ! -f /sgl-workspace/aiter/aiter/jit/module_rmsnorm.so ]; then
    echo "Restoring aiter JIT cache from NFS..."
    cp /shared_nfs/xiaofei/aiter_jit_cache/*.so /sgl-workspace/aiter/aiter/jit/ 2>/dev/null || true
    cp -r /shared_nfs/xiaofei/aiter_jit_cache/build /sgl-workspace/aiter/aiter/jit/ 2>/dev/null || true
fi

# 3. 清理 aiter JIT 锁
find /sgl-workspace/aiter/aiter/jit/build -name "lock*" -exec rm -f {} \; 2>/dev/null || true

# 4. 设置环境变量
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export SGLANG_DISABLE_FA3=1
export NCCL_MIN_NCHANNELS=112
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
export NCCL_CROSS_NIC=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export HSA_NO_SCRATCH_RECLAIM=1

# 5. 启动训练
cd /shared_nfs/xiaofei/verl
MODEL_PATH=${MODEL_PATH:-/shared_nfs/xiaofei/Primus/output/qwen3_8b_safe_hf}
GPUS_PER_NODE=8

PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/shared_nfs/xiaofei/verl/data/gsm8k/train.parquet \
    data.val_files=/shared_nfs/xiaofei/verl/data/gsm8k/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=verl_grpo_qwen3_8b \
    trainer.experiment_name=qwen3_8b_grpo_gsm8k \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=${NNODES:-1} \
    trainer.save_freq=100 \
    trainer.default_local_dir=/shared_nfs/xiaofei/verl_checkpoints \
    trainer.test_freq=5 \
    trainer.total_epochs=2
```

---

## 多节点（Multi-Node）训练验证总结

### 验证环境

| 项目 | 配置 |
|------|------|
| 节点数 | 2 |
| GPU | AMD MI355X × 8/node（共 16 GPU） |
| 网络 | AINIC（ionic RoCEv2 RDMA） |
| 集群管理 | Ray 2.44.1（Kubernetes RayJob） |
| 存储 | 共享 NFS（`/shared_nfs`） |
| 模型 | Qwen3-8B（SFT 后） |
| 数据集 | GSM8K |

### 最终训练成果

| 指标 | FSDP2 | FSDP1 |
|------|-------|-------|
| 总训练时间 | 31 分 51 秒（58 步，2 epochs） | 29 分 11 秒 |
| 平均步时间 | ~31s/step | ~28s/step |
| 吞吐量 | ~1,490 tok/s | ~1,665 tok/s |
| GSM8K 验证准确率 | 5.76% → **65.66%** | 6.37% → **63.08%** |
| 显存占用（allocated/reserved） | 25.1GB / 34.6GB | 25.3GB / 41.6GB |

> 注：ref 模型已关闭 `param_offload` 和 `reshard_after_forward`，详见问题 5。

### 遇到的问题与解决方案

#### 问题 1：`KeyError: 'CUDA_VISIBLE_DEVICES'`（多节点 Ray worker）

**现象**：通过 `ray job submit` 提交训练后，sglang rollout 初始化时在 `async_sglang_server.py` 第 455 行崩溃。

**原因**：原始代码直接使用 `os.environ[visible_devices_keyword]` 访问环境变量。在多节点 Ray worker 中，`CUDA_VISIBLE_DEVICES` 可能未被显式设置（尤其在设置了 `RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1` 后），导致 KeyError。

**解决方案（二选一）**：

- **方案 A（简单，不改源码）**：在启动脚本中显式设置 `export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`，并通过 `runtime_env.env_vars` 确保传递到 Ray worker 进程中。只要 Ray worker 内能拿到这个变量就不会报错。
- **方案 B（稳健，改源码）**：修改 `async_sglang_server.py`，将 `os.environ[visible_devices_keyword]` 改为 `os.environ.get()` 加 fallback 链，防御环境变量传播不到位的情况：

```python
def _get_worker_info(self):
    node_id = ray.get_runtime_context().get_node_id()
    devices = os.environ.get(visible_devices_keyword,
                os.environ.get("HIP_VISIBLE_DEVICES",
                os.environ.get("ROCR_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")))
    return (node_id, devices)
```

> 我们实际采用的是方案 B + 同时设置 `CUDA_VISIBLE_DEVICES`（双保险）。如果不想改源码，只用方案 A 也可以。

#### 问题 2：Ray 版本不匹配

**现象**：`RuntimeError: Ray version mismatch: cluster has Ray version 2.44.1 but local Ray version is 2.54.0`

**原因**：`pip install -e .` 安装 verl 时，其依赖自动安装了较新的 Ray Python 包（2.54.0），与 KubeRay RayJob 镜像中运行的 Ray 集群版本（2.44.1）不匹配。注意这里的版本是指 Ray 集群的 Ray 版本（由 RayJob 的 pod 镜像决定），不是 KubeRay Operator 的版本（1.4.2）。

**解决方案**：安装 verl 后，将 Ray 降级到与集群一致的版本：

```bash
# 先查看集群中的 Ray 版本
ray --version  # 或 python -c "import ray; print(ray.__version__)"

# 降级到一致版本
pip install ray==2.44.1
```

**教训**：在已有 Ray 集群的节点上安装 verl 前，先记录当前 Ray 版本，安装后检查并恢复。

#### 问题 3：多节点训练极慢（270s/step vs 29s/step）

**现象**：首次多节点训练启动后，每步耗时约 270 秒（单节点仅 29 秒），NCCL 日志显示 `Could not find any local path from gpu 2 to net`。

**原因**：AINIC 驱动未安装，RCCL 只能通过 Kubernetes overlay 网络（TCP socket）做跨节点通信，带宽极低。

**解决方案**：在两个节点上都安装 AINIC 驱动：

```bash
export AINIC_DRIVER_VERSION=1.117.5-a-56
apt-get update && apt-get install -y jq initramfs-tools
bash /shared-data/build_ainic.sh
```

#### 问题 4：NCCL IB GID Index 错误

**现象**：AINIC 安装后跨节点 NCCL 测试仍然失败：

```
Call to ibv_modify_qp failed with 61 No data available,
on dev ionic_0:0, local GID index 3, local GID N/A
```

**原因**：官方文档设置 `NCCL_IB_GID_INDEX=3`，但 AINIC `ionic` 设备的 GID 表中 index 3 为空。通过检查发现只有 index 0 和 1 有效。

**分析方法**：

```bash
cat /sys/class/infiniband/ionic_0/ports/1/gids/0  # fe80::... (link-local)
cat /sys/class/infiniband/ionic_0/ports/1/gids/1  # ::ffff:10.x.x.x (RoCEv2)
cat /sys/class/infiniband/ionic_0/ports/1/gids/2  # (empty)
cat /sys/class/infiniband/ionic_0/ports/1/gids/3  # (empty)
```

**解决方案**：

```bash
export NCCL_IB_GID_INDEX=1
```

**教训**：不要盲目使用官方文档的 GID index，必须检查实际硬件的 GID 表。AINIC 与 Mellanox IB 卡的 GID 表结构不同。

#### 问题 5：FSDP2 多节点 ref 模型计算极慢（538-1008s vs 正常 1.3s）

**现象**：多节点 FSDP2 训练中，`timing_s/ref`（reference model 前向推理）耗时 538-1008 秒，而同配置 FSDP1 仅需 2-69 秒，单节点 FSDP2 也只需几秒。

**原因**：`verl/workers/fsdp_workers.py` 中，ref 模型在 FSDP2 下的 `CPUOffloadPolicy` 和 `reshard_after_forward=True` 是**硬编码**的，不受配置控制：

```python
# FSDP2 路径（原始代码 L614-615）
cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

# FSDP1 路径（原始代码 L589）
cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
```

ref 模型是 eval-only（只做前向推理），开启 offload+reshard 导致每个 micro-batch forward 都要：
1. 跨节点 all-gather 参数（RDMA ~40 GB/s，远慢于节点内 xGMI ~800 GB/s）
2. 前向后 reshard 回分片状态
3. 下一个 micro-batch 再次 all-gather

这在单节点内可接受（xGMI 快），但多节点下成为严重瓶颈。

**解决方案**：

1. **修改源码** `verl/workers/fsdp_workers.py`，让 ref 的 offload 受配置控制：

```python
# FSDP2 路径修复（替换 L610-615）
if role == "actor" and fsdp_config.offload_policy:
    cpu_offload = CPUOffloadPolicy(pin_memory=True)
    self._is_offload_param = False
    self._is_offload_optimizer = False
elif role == "actor":
    cpu_offload = None
else:
    cpu_offload = CPUOffloadPolicy(pin_memory=True) if fsdp_config.param_offload else None

# FSDP1 路径修复（替换 L589）
if role == "actor":
    cpu_offload = None
else:
    cpu_offload = CPUOffload(offload_params=True) if fsdp_config.param_offload else None
```

2. **训练配置**中关闭 ref 的 offload 和 reshard：

```bash
actor_rollout_ref.ref.fsdp_config.param_offload=False
actor_rollout_ref.ref.fsdp_config.reshard_after_forward=False
```

**性能对比**：

| 配置 | timing_s/ref | timing_s/step | throughput |
|------|-------------|---------------|------------|
| FSDP2 + ref offload+reshard（原始） | 538-1008s | ~1050s | ~47 tok/s |
| FSDP1（对照） | 2-69s | 29-96s | 488-1,423 tok/s |
| **FSDP2 关闭 ref offload+reshard** | **~1.3s** | **~31s** | **~1,490 tok/s** |
| **FSDP1 关闭 ref offload+reshard** | **~1.3s** | **~28s** | **~1,665 tok/s** |

**适用条件**：关闭 ref offload+reshard 后，ref 参数常驻 GPU。对 Qwen3-8B 在 16 卡上，每卡额外占用约 1GB bf16 参数，显存安全（25.1GB/192GB）。对于更大模型（如 70B），需评估显存是否足够。

#### 问题 6：`ray status` 显示 GPU 使用率低

**现象**：`rocm-smi` 显示 100% GPU 利用率，但 `ray status` 仅显示 5.33/16 GPU。

**原因**：Ray 的资源计数基于直接请求 GPU 资源的 actor 数量。sglang rollout server 作为子进程启动，不通过 Ray 分配 GPU，所以 Ray 看不到它们。

**结论**：`rocm-smi` 是判断实际利用率的正确工具。

### 多节点启动方式

#### 方式 1：在 Ray head 节点直接运行脚本

如果已经 SSH 到 Ray head 节点且集群已就绪：

```bash
cd /shared_nfs/xiaofei/verl
bash run_multinode_grpo.sh
```

#### 方式 2：通过 Ray Job Submit 提交（推荐）

从任意能访问 Ray Dashboard 的节点提交，不需要 SSH 到 head 节点：

```python
from ray.job_submission import JobSubmissionClient
client = JobSubmissionClient("http://<head-node-ip>:8265")
job_id = client.submit_job(
    entrypoint="bash /shared_nfs/xiaofei/verl/run_multinode_grpo.sh",
    runtime_env={
        "working_dir": "/shared_nfs/xiaofei/verl",
        "env_vars": {"PYTHONUNBUFFERED": "1"}
    }
)
print("Job ID:", job_id)
```

监控进度：

```python
print(client.get_job_status(job_id))
print(client.get_job_logs(job_id))
```

#### 多节点训练脚本（`run_multinode_grpo.sh`）

```bash
#!/bin/bash
set -e

cd /shared_nfs/xiaofei/verl

# GPU 设备
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Ray 设备变量保护（Ray < 2.45.0 用 ROCR，>= 2.45.0 用 HIP）
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export SGLANG_DISABLE_FA3=1

# AINIC / InfiniBand 配置（平台已安装驱动时生效）
export NCCL_IB_HCA=ionic_0,ionic_2,ionic_3,ionic_4,ionic_5,ionic_7,ionic_8,ionic_9
export NCCL_IB_GID_INDEX=1
export NCCL_MIN_NCHANNELS=112
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
export NCCL_CROSS_NIC=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false
export HSA_NO_SCRATCH_RECLAIM=1
export LD_LIBRARY_PATH=/usr/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

NNODES=${NNODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
SAVE_DIR=${SAVE_DIR:-/shared_nfs/xiaofei/verl_checkpoints_multinode}

# 学习率按 linear scaling rule：lr = base_lr × NNODES
# 单节点 base_lr=1e-6，2 节点 lr=2e-6
LR=$(python3 -c "print(1e-6 * $NNODES)")

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/shared_nfs/xiaofei/verl/data/gsm8k/train.parquet \
    data.val_files=/shared_nfs/xiaofei/verl/data/gsm8k/test.parquet \
    data.train_batch_size=$((128 * NNODES)) \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=/shared_nfs/xiaofei/Primus/output/qwen3_8b_safe_hf \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((64 * NNODES)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=False \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=verl_grpo_qwen3_8b_multinode \
    trainer.experiment_name=qwen3_8b_grpo_${NNODES}node \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=50 \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.test_freq=5 \
    trainer.total_epochs=2
```

#### 多节点参数扩展规则

| 参数 | 单节点 (1 node) | N 节点 | 规则 |
|------|----------------|--------|------|
| `trainer.nnodes` | 1 | N | 节点数 |
| `data.train_batch_size` | 128 | 128 × N | 按节点数线性扩大 |
| `actor.ppo_mini_batch_size` | 64 | 64 × N | 按节点数线性扩大 |
| `actor.optim.lr` | 1e-6 | 1e-6 × N | Linear Scaling Rule |
| `ppo_micro_batch_size_per_gpu` | 4 | 4 | 不变（每 GPU 负载不变） |
| `rollout.tensor_model_parallel_size` | 8 | 8 | 不变（每节点内 TP） |
| `rollout.free_cache_engine` | 未设 | True | 多节点建议开启 |

> **Linear Scaling Rule**：batch size 扩大 N 倍时，学习率也应扩大 N 倍，以保持相同的有效更新幅度。这是分布式训练的标准做法（参考 Goyal et al., "Accurate, Large Minibatch SGD"）。

### 多节点 vs 单节点性能对比

| 指标 | 单节点 (8 GPU) | 多节点无 AINIC (16 GPU) | 多节点有 AINIC + ref offload (16 GPU) | 多节点有 AINIC + 关闭 ref offload (16 GPU) |
|------|---------------|------------------------|--------------------------------------|------------------------------------------|
| 步时间 | ~29s | ~270s | ~1050s | **~28-31s** |
| 吞吐量 | ~1630 tok/s | ~180 tok/s | ~47 tok/s | **~1,490-1,665 tok/s** |
| timing_s/ref | ~2s | — | 538-1008s | **~1.3s** |
| 加速比 | 1.0x | 0.1x | 0.03x | **~1.0x** |

关闭 ref 的 offload+reshard 后，多节点性能与单节点持平。AINIC RDMA 确保了跨节点通信带宽（~40 GB/s），但必须避免 ref 模型在 eval mode 下反复跨节点 all-gather。

### 代码修改清单

| 文件 | 修改 | 影响范围 |
|------|------|---------|
| `async_sglang_server.py` | `_is_rocm()` + `enable_memory_saver: False` + aiter backend | 单节点 + 多节点 |
| `async_sglang_server.py` | `os.environ.get()` fallback（或设置 `CUDA_VISIBLE_DEVICES`） | 多节点 |
| `fsdp_workers.py` | ref 模型 CPU offload 受 `fsdp_config.param_offload` 控制（FSDP1 + FSDP2） | 多节点（单节点影响小） |

### 多节点训练脚本

推荐使用 `rl-scripts/` 目录下的脚本：

| 脚本 | 策略 | 说明 |
|------|------|------|
| `rl-scripts/run_multinode_grpo_fsdp2_no_reshard_offload.sh` | FSDP2 | 关闭 ref offload+reshard，已验证 |
| `rl-scripts/run_multinode_grpo_fsdp1.sh` | FSDP1 | 关闭 ref offload+reshard，已验证 |

### param_offload 与 reshard_after_forward 使用建议

这两个参数对 **actor 模型**（有梯度更新）仍然推荐开启，因为 actor 需要在训练和 rollout 之间切换，offload 可以为 sglang 腾出显存。

对 **ref 模型**（eval-only），是否开启取决于显存约束：

| 场景 | ref offload | ref reshard | 说明 |
|------|-------------|-------------|------|
| 显存充足（如 8B 模型在 MI355X 192GB） | False | False | 推荐，性能最优 |
| 显存紧张（如 70B+ 模型或小显存卡） | True | True | 牺牲性能换显存 |
| 单节点 | 影响小 | 影响小 | 节点内 xGMI 带宽高，offload+reshard 开销可接受 |

