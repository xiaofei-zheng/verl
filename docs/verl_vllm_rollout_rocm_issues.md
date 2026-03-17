# verl + FSDP + vLLM Rollout 在 ROCm 环境下的问题与解决方案

## 环境信息

- GPU: AMD MI355X × 8 (288GB VRAM each)
- 平台: ROCm (HIP)
- 镜像: `rocm/vllm:latest`
- 框架: verl (ByteDance RL training framework)
- 训练后端: FSDP
- 推理后端: vLLM rollout
- 模型: Qwen3-8B

## 当前状态

**未成功**。由于 CUDA IPC 兼容性问题，vLLM rollout 在 ROCm 上无法正常工作。已部分修复但训练进程 hang，最终转用 sglang rollout 成功。

本文档记录遇到的问题和尝试的修复，供后续 vLLM ROCm 兼容性改善后参考。

---

## 问题 1: `TypeError: 'str' object is not callable` (CUDA IPC 权重同步)

### 现象

训练启动后，在 vLLM rollout 的权重同步阶段报错：

```python
TypeError: 'str' object is not callable
```

错误发生在 `verl/workers/rollout/vllm_rollout/utils.py` 的 `rebuild_ipc` 函数（第 119 行）。

### 原因

verl 使用 PyTorch 的 `torch.multiprocessing.reductions.reduce_tensor()` 做 CUDA IPC handle 序列化，将 actor 训练好的权重传递给 vLLM 推理引擎。该函数返回一个 `(func, args)` tuple：

- **NVIDIA (CUDA)**: `func` 是一个可调用的函数（如 `rebuild_cuda_tensor`），`func(*args)` 可以重建 tensor
- **ROCm (HIP)**: `func` 变成了一个**字符串**（函数名），无法直接调用

这是 PyTorch ROCm 版本在 `cuda_ipc` 实现上的根本性差异。

### 尝试的修复

修改 `verl/utils/device.py` 中的 `is_support_ipc()` 函数，让 ROCm 环境返回 `False`，强制使用 shared memory 替代 CUDA IPC：

```python
# verl/utils/device.py
def is_support_ipc():
    if is_cuda_available:
        import torch
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return False  # ROCm 不支持 CUDA IPC
        return True
    return False
```

### 修复效果

修改后 verl 使用 shared memory 方式传输权重，绕过了 CUDA IPC。但训练进程在后续阶段 **hang 住不动**（进程存活但无输出），具体原因未排查清楚。

### 修改文件

- `verl/utils/device.py` 中的 `is_support_ipc()` 函数

### 备注

如果后续 ROCm 版 PyTorch 修复了 `reduce_tensor` 的 IPC handle 格式（使 `func` 返回 callable），这个问题应该自然解决，不需要修改 verl 代码。

> **重要**: 如果从 sglang 切回 vLLM，记得恢复此修改（`is_support_ipc` 改回 ROCm 返回 `False`），因为 sglang rollout 不依赖这个函数。

---

## 问题 2: 进程 hang（shared memory 模式下）

### 现象

在修复了 CUDA IPC 问题（使用 shared memory）后，训练进程能启动到 rollout 阶段，但随后 **hang 住**：

- 进程存活（`ps` 可见）
- GPU 利用率时而 100% 时而 0%
- 无新的日志输出
- 无报错信息

### 可能原因

1. shared memory 传输 8B 模型权重可能有 deadlock
2. vLLM 的某些内部操作（如 CUDA graph capture）在 ROCm 上不兼容
3. RCCL 通信在 shared memory + vLLM worker 之间的协调有问题

### 状态

**未解决**。由于此问题难以调试，转用 sglang rollout 成功后未继续排查。

---

## 问题 3: 环境变量设置（与 sglang 共同问题）

### 现象

与 sglang 类似的环境变量问题：

- `HIP_VISIBLE_DEVICES` 和 `ROCR_VISIBLE_DEVICES` 冲突
- verl/Ray 对 GPU 设备的管理逻辑

### 解决方案

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
```

---

## 如果后续要重试 vLLM Rollout

### 前置检查

1. 确认 PyTorch ROCm 版本的 `reduce_tensor()` 是否已修复 IPC handle 格式
2. 确认 vLLM ROCm 版本是否支持 `enforce_eager=True` 模式的完整推理
3. 确认 RCCL shared memory 在多 worker 场景下是否稳定

### 需要的代码修改

```
verl/utils/device.py
  └─ is_support_ipc(): ROCm 环境返回 False（如果 IPC 仍不兼容）
```

### 建议的启动参数

```bash
python3 -m verl.trainer.main_ppo \
    ...
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    ...
```

关键点：
- `enforce_eager=True`：避免 CUDA graph capture 在 ROCm 上的兼容性问题
- `tensor_model_parallel_size=8`：避免多 vLLM server 竞争显存（与 sglang 相同问题）
- `param_offload=True`：为 vLLM 释放 GPU 显存

---

## 与 sglang 的对比

| 维度 | vLLM | sglang |
|------|------|--------|
| ROCm 支持 | 部分（IPC 不兼容） | 完整（aiter backend） |
| 权重同步 | CUDA IPC / shared memory | sglang 自有机制 |
| 在 verl 中的状态 | hang | 成功训练 |
| 推荐度 | 暂不推荐 | 推荐 |
