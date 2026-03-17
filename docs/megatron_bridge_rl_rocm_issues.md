# verl + Megatron-Bridge RL 训练在 ROCm 上的问题记录

> 环境: ROCm MI355X, `rocm/primus:v26.1` pod (后替换为 vLLM 镜像), Qwen3-8B, verl GRPO
> 日期: 2026-02-24 ~ 2026-02-25
> 结论: **Megatron-Bridge 后端在 ROCm 上跑 RL 训练目前不可行**, 建议使用 FSDP 后端

## 1. transformer_engine 依赖问题

**错误**: `ModuleNotFoundError: No module named 'transformer_engine'`

**原因**: Megatron-Core 的 `gpt_layer_specs.py` 默认尝试 import `transformer_engine` (NVIDIA 专属库), ROCm 上不可用。

**尝试的解决方案**:
- 创建 mock `transformer_engine` 包 → **失败**, Megatron-Core 的 `extensions/transformer_engine.py` 需要继承 TE 的真实类
- 修改 `gpt_layer_specs.py` 的异常处理: `except ImportError` → `except (ImportError, Exception)`, 并设 `TENorm = None` → **成功**

**修改文件**: `Primus/third_party/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/models/gpt/gpt_layer_specs.py`

## 2. TENorm 未定义

**错误**: `NameError: name 'TENorm' is not defined`

**原因**: `HAVE_TE=False` 时, 代码仍然走 `if use_transformer_engine:` 分支, 尝试使用 `TENorm`。

**解决方案**: 修改条件为 `if use_transformer_engine and HAVE_TE:`, 确保只在 TE 真正可用时才使用。

**修改文件**: 同上, 两处 (约第 620 行和第 771 行)

## 3. RMSNorm 不兼容 FusedLayerNorm

**错误**: `AssertionError: (RMSNorm) is not supported in FusedLayerNorm`

**原因**: Qwen3 使用 RMSNorm, 但 fallback 路径使用的 `LNImpl` = `FusedLayerNorm` (Apex), 只支持 LayerNorm。

**解决方案**: 在 `else` 分支中检测 `normalization` 类型, 如果不是 LayerNorm 则使用 `WrappedTorchNorm`:
```python
norm_type = normalization or getattr(config, 'normalization', None)
if norm_type and norm_type != "LayerNorm":
    from megatron.core.transformer.torch_norm import WrappedTorchNorm
    layer_norm_impl = WrappedTorchNorm
else:
    layer_norm_impl = LNImpl
```

**修改文件**: 同上

## 4. WrappedTorchNorm 不支持 sequence_parallel

**错误**: `AssertionError: sequence parallel not supported by torch LayerNorm`

**原因**: `WrappedTorchNorm` 不支持 sequence parallelism。

**解决方案**: 通过 Hydra 配置禁用:
```
+actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=false
+actor_rollout_ref.ref.megatron.override_transformer_config.sequence_parallel=false
```

## 5. mbridge 权重映射缺失 (local spec layernorm)

**错误**: `NotImplementedError: Unsupported parameter name: decoder.layers.0.input_layernorm.weight` 和 `decoder.layers.0.pre_mlp_layernorm.weight`

**原因**: 当使用 `transformer_impl=local` 时, layernorm 权重名称与 TE 模式不同:
- TE 模式: `self_attention.linear_qkv.layer_norm_weight` (fused)
- Local 模式: `input_layernorm.weight`, `pre_mlp_layernorm.weight` (独立)

`mbridge` 的 `Qwen2Bridge` 没有这些 local spec 权重名称的映射。

**尝试的解决方案**: 在 `mbridge/models/qwen2.py` 中添加 `_LOCAL_LAYERNORM_MAPPING` 和重写 `_weight_name_mapping_mcore_to_hf` → **部分成功**, `input_layernorm.weight` 修好了, 但 `pre_mlp_layernorm.weight` 仍然出错, 因为父类的 MLP 映射逻辑会优先捕获包含某些关键字的权重名称。

**修改文件**: `/usr/local/lib/python3.12/dist-packages/mbridge/models/qwen2.py`

## 6. NVTE 环境变量冲突

**错误**: `AssertionError: NVTE_FUSED_ATTN set to 1`

**原因**: Ray worker 之间 `NVTE_*` 环境变量设置不一致。

**解决方案**:
```bash
unset NVTE_FUSED_ATTN
unset NVTE_FLASH_ATTN
unset NVTE_UNFUSED_ATTN
```
并显式设置:
```
+actor_rollout_ref.ref.megatron.override_transformer_config.attention_backend=flash
```

## 7. ROCm 上 CUDA IPC 不兼容

**错误**: `TypeError: 'str' object is not callable` (在 `rebuild_ipc` 中)

**原因**: verl 使用 PyTorch 的 `reduce_tensor` 做 CUDA IPC handle 序列化, ROCm (HIP) 上返回的 handle 格式不同, `func` 字段是字符串而非可调用对象。

**解决方案**: 修改 `verl/utils/device.py` 的 `is_support_ipc()`, 在 ROCm 上返回 `False`, 使用 shared memory 替代 IPC:
```python
if is_cuda_available:
    import torch
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return False
    return True
```

**修改文件**: `verl/verl/utils/device.py`

## 8. 最终状态: 进程卡住

修复 IPC 问题后, 训练进程启动成功, vLLM rollout 引擎初始化完成 (CUDA graphs captured), 但进程在权重同步或 rollout 阶段卡住, 约 15 分钟无新输出。可能与 shared memory 传输大模型权重的性能或 ROCm 平台兼容性有关。

## 总结

Megatron-Bridge 后端在 ROCm 上面临的核心问题是对 NVIDIA `transformer_engine` 库的深度依赖。虽然可以通过 `transformer_impl=local` 绕过, 但会引发一系列连锁问题 (layernorm 映射、序列并行、IPC 兼容性)。每解决一个问题都会暴露下一个, 而且最终仍然卡住。

**建议**: 在 ROCm 上使用 verl 的 **FSDP 后端** 进行 RL 训练, 它不依赖 Megatron-Core/transformer_engine, 更适合 ROCm 平台。
