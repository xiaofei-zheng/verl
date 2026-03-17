# verl + FSDP + SGLang Rollout 在 ROCm 环境下的问题与解决方案

## 环境信息

- GPU: AMD MI355X × 8 (288GB VRAM each)
- 平台: ROCm (HIP)
- 镜像: `rocm/sglang` (基于 sglang 最新版)
- 框架: verl (ByteDance RL training framework)
- 训练后端: FSDP
- 推理后端: sglang rollout
- 模型: Qwen3-8B

## 前置条件

sglang 独立推理（`python -m sglang.launch_server`）可以正常工作。以下问题均出现在 **verl 通过 Ray 调用 sglang** 时。

---

## 问题 1: `KeyError: 'CUDA_VISIBLE_DEVICES'`

### 现象

sglang rollout 初始化时报错：

```
KeyError: 'CUDA_VISIBLE_DEVICES'
```

### 原因

ROCm 环境下通常只设 `HIP_VISIBLE_DEVICES`，不设 `CUDA_VISIBLE_DEVICES`。但 sglang 或底层库直接读取 `os.environ['CUDA_VISIBLE_DEVICES']`。

### 解决方案

verl 内部会将 `HIP_VISIBLE_DEVICES` 映射为 `CUDA_VISIBLE_DEVICES`，所以只需要设置 `HIP_VISIBLE_DEVICES`，不要同时设置 `ROCR_VISIBLE_DEVICES`：

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
```

> verl 的 `verl/single_controller/base/worker.py` 会自动将 `HIP_VISIBLE_DEVICES` 映射到 `CUDA_VISIBLE_DEVICES`，如果同时设置了 `ROCR_VISIBLE_DEVICES` 会报 `ValueError`。

---

## 问题 2: `ROCR_VISIBLE_DEVICES` 与 `HIP_VISIBLE_DEVICES` 冲突

### 现象

```
ValueError: Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set.
```

### 原因

verl 内部（`worker.py`）检测到 `HIP_VISIBLE_DEVICES` 后会自行设置 `CUDA_VISIBLE_DEVICES`，如果 `ROCR_VISIBLE_DEVICES` 也被设置就会报错。

### 解决方案

只设置 `HIP_VISIBLE_DEVICES`，不要设置 `ROCR_VISIBLE_DEVICES`：

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ROCR_VISIBLE_DEVICES
```

---

## 问题 3: `libcuda.so.1: cannot open shared object file`

### 现象

sglang 初始化时报错：

```
OSError: libcuda.so.1: cannot open shared object file: No such file or directory
```

### 原因

verl 在 `async_sglang_server.py` 中硬编码了 `enable_memory_saver: True`。`torch_memory_saver` 这个包在模块级别导入时会尝试加载一个链接到 NVIDIA `libcuda.so.1` 的 C 动态库，ROCm 环境下不存在此文件。

### 解决方案

修改 `verl/workers/rollout/sglang_rollout/async_sglang_server.py`：

```python
# 原始:
"enable_memory_saver": True,

# 修改为:
"enable_memory_saver": False,
```

### 修改文件

- `verl/workers/rollout/sglang_rollout/async_sglang_server.py` (第 ~185 行)

---

## 问题 4: `ImportError: Can not import FA3 in sgl_kernel`

### 现象

```
ImportError: Can not import FA3 in sgl_kernel. Please check your installation.
```

即使设置了 `enforce_eager=True` 和 `SGL_DISABLE_FA3=1` 也不行。

### 原因

verl 在 `async_sglang_server.py` 中硬编码了 attention backend 为 `"fa3"`：

```python
"mm_attention_backend": "fa3",
"attention_backend": attention_backend if attention_backend is not None else "fa3",
```

Flash Attention 3 (FA3) 是 NVIDIA Hopper 架构专用的，ROCm 不支持。ROCm 应该使用 `"aiter"` backend。

### 解决方案

修改 `verl/workers/rollout/sglang_rollout/async_sglang_server.py`，根据平台选择 backend：

```python
def _is_rocm() -> bool:
    return hasattr(torch.version, 'hip') and torch.version.hip is not None

# 在构建 args 时:
"mm_attention_backend": "fa3" if not _is_rocm() else "aiter",
"attention_backend": attention_backend if attention_backend is not None else ("aiter" if _is_rocm() else "fa3"),
```

同时设置环境变量（以防其他路径仍尝试导入 FA3）：

```bash
export SGLANG_DISABLE_FA3=1
```

> 注意: `SGL_DISABLE_FA3` 已被废弃，使用 `SGLANG_DISABLE_FA3`。

### 修改文件

- `verl/workers/rollout/sglang_rollout/async_sglang_server.py` (顶部添加 `_is_rocm()` 函数，第 ~198-199 行修改 backend)

---

## 问题 5: `RuntimeError: Not enough memory` (sglang KV cache 初始化)

### 现象

```
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
Current value: self.server_args.mem_fraction_static=0.34
```

MI355X 有 288GB VRAM，8B 模型不可能 OOM。

### 原因

verl 用 `tensor_model_parallel_size` 和 `data_parallel_size` 控制启动多少个独立的 sglang server。当 TP=2, DP=1 时，8 GPU 会启动 **4 个独立的 sglang server**（每个 TP=2）。这些 server 并发初始化，先启动的 server 分配 KV cache 和 CUDA graph 占用大量显存，后启动的 server 在 `init_memory_pool` 中检测到可用显存不足就报错。

具体来说，sglang 的 `profile_max_num_token` 计算方式：

```python
rest_memory = available_gpu_memory - total_gpu_memory * (1 - mem_fraction_static)
```

`total_gpu_memory` 用的是 GPU 总显存（287GB），当 `available_gpu_memory` 因为其他 server 先占了而降到 164GB 时：
- `rest_memory = 164 - 287 * (1 - 0.34) = 164 - 189 = -25` → 负数 → 报错

### 解决方案

设置 `tensor_model_parallel_size=8`，使所有 GPU 属于同一个 sglang server，避免多 server 竞争显存：

```bash
actor_rollout_ref.rollout.tensor_model_parallel_size=8
```

同时开启 FSDP 的 offload 为 sglang 释放显存：

```bash
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
```

> 注意: TP=8 对 8B 模型来说有些浪费（单卡就能推理），但在 verl 的架构下这是确保 sglang 正常初始化的简单方案。

---

## 最终可用的启动命令

```bash
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

MODEL_PATH=/shared_nfs/xiaofei/Primus/output/qwen3_8b_safe_hf
GPUS_PER_NODE=8

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/gsm8k/train.parquet \
    data.val_files=data/gsm8k/test.parquet \
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
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
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
    trainer.experiment_name=qwen3_8b_grpo_gsm8k_fsdp_sglang \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=3
```

## 代码修改清单

| 文件 | 修改内容 |
|------|---------|
| `verl/workers/rollout/sglang_rollout/async_sglang_server.py` | 添加 `_is_rocm()` 函数; `enable_memory_saver` 改为 `False`; `attention_backend` 和 `mm_attention_backend` 根据 ROCm 选 `"aiter"` |

## 训练性能参考 (Qwen3-8B, 8× MI355X)

| 指标 | 值 |
|------|-----|
| 每步耗时 | ~29-33s |
| 吞吐量 | ~1620 tokens/s |
| GPU 显存使用 | ~15.6GB allocated / 30.1GB reserved |
| 预估总训练时间 (3 epochs) | ~4.5 小时 |
