#!/bin/bash
# Qwen3-235B-A22B GRPO RL — 8-node x 4-GPU Megatron + sglang
# Reduces workers per node from 8→4 to avoid cgroup OOM during checkpoint save.
# On ROCm, HIP maps GPU memory into process RSS; 8 workers x 350GB = 2800GB = cgroup limit.
# With 4 workers/node: 4 x 350GB = 1400GB, leaving 1400GB headroom for save.
#
# Self-contained: includes source code patches, can be submitted directly as a rayjob.
set -xeuo pipefail

cd /shared_nfs/xiaofei/verl

# ============================================================
# Source code patches (applied idempotently)
# ============================================================

# Patch 1: enable_memory_saver reads from config instead of hardcoded False
python3 - <<'PATCH1'
import pathlib
f = pathlib.Path("verl/workers/rollout/sglang_rollout/async_sglang_server.py")
src = f.read_text()
old = '"enable_memory_saver": False,'
new = '"enable_memory_saver": self.config.get("free_cache_engine", False),'
if old in src:
    f.write_text(src.replace(old, new))
    print("[patch] async_sglang_server.py: enable_memory_saver patched")
elif new in src:
    print("[patch] async_sglang_server.py: already patched")
else:
    print("[patch] WARNING: pattern not found in async_sglang_server.py")
PATCH1

# Patch 2: CPU-based checkpoint save (avoid GPU allocation that inflates RSS on ROCm)
python3 - <<'PATCH2'
import pathlib
f = pathlib.Path("verl/workers/megatron_workers.py")
src = f.read_text()
if "_swap_params_to_cpu_for_save" in src:
    print("[patch] megatron_workers.py: CPU-save already patched")
elif "load_megatron_model_to_gpu(self.actor_module)" in src and "def save_checkpoint" in src:
    old_block = '''    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)'''
    new_block = '''    def _swap_params_to_cpu_for_save(self):
        """Swap model params from empty GPU views to CPU data for saving without GPU alloc."""
        from megatron.core.distributed import DistributedDataParallel as MCoreDDP
        saved_params = []
        for model_chunk in self.actor_module:
            if not isinstance(model_chunk, MCoreDDP):
                continue
            for buffers in [model_chunk.buffers, model_chunk.expert_parallel_buffers]:
                for buffer in buffers:
                    if buffer.param_data.storage().size() > 0:
                        continue
                    if not hasattr(buffer.param_data, "cpu_data"):
                        continue
                    cpu_data = buffer.param_data.cpu_data
                    buf_offset = buffer.param_data.storage_offset()
                    for param in buffer.params:
                        orig_view = param.data
                        param_offset = orig_view.storage_offset() - buf_offset
                        cpu_view = cpu_data[param_offset:param_offset + param.numel()].view(orig_view.shape)
                        saved_params.append((param, orig_view))
                        param.data = cpu_view
        return saved_params

    def _restore_params_after_save(self, saved_params):
        for param, orig_view in saved_params:
            param.data = orig_view

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        saved_params = None
        if self._is_offload_param:
            saved_params = self._swap_params_to_cpu_for_save()'''
    if old_block in src:
        src = src.replace(old_block, new_block)
        # Also patch the post-save: remove offload_megatron_model_to_cpu, add restore
        old_post = '''        torch.distributed.barrier()
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)'''
        new_post = '''        torch.distributed.barrier()
        if saved_params is not None:
            self._restore_params_after_save(saved_params)'''
        if old_post in src:
            src = src.replace(old_post, new_post)
        f.write_text(src)
        print("[patch] megatron_workers.py: CPU-save patched")
    else:
        print("[patch] WARNING: could not find save_checkpoint pattern in megatron_workers.py")
else:
    print("[patch] megatron_workers.py: unrecognized state, skipping")
PATCH2

# Patch 3: skip sharding validation on ROCm (known incompatibility)
python3 - <<'PATCH3'
import pathlib
f = pathlib.Path("verl/utils/megatron/dist_checkpointing.py")
src = f.read_text()
old = "validate_sharding_integrity = True"
new = 'is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None\n    validate_sharding_integrity = not is_rocm'
if old in src:
    f.write_text(src.replace(old, new))
    print("[patch] dist_checkpointing.py: ROCm sharding validation patched")
elif "not is_rocm" in src:
    print("[patch] dist_checkpointing.py: already patched")
else:
    print("[patch] dist_checkpointing.py: pattern not found, skipping")
PATCH3

echo "[pre-flight] Cleaning __pycache__ ..."
find /shared_nfs/xiaofei/verl/verl -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "[pre-flight] __pycache__ cleaned"

# ---- GPU visibility ----
# All 8 GPUs visible; n_gpus_per_node=4 ensures only 4 workers per node
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1

# ---- Ray timeouts (large model needs long init) ----
export RAY_HEALTH_CHECK_PERIOD_MS=600000
export RAY_HEALTH_CHECK_TIMEOUT_MS=1800000
export RAY_GRPC_KEEPALIVE_TIME_MS=300000
export RAY_GRPC_KEEPALIVE_TIMEOUT_MS=1800000
export RAY_grpc_server_keepalive_time_ms=300000
export RAY_grpc_server_keepalive_timeout_ms=1800000
export RAY_gcs_server_request_timeout_seconds=1800
export RAY_object_timeout_milliseconds=1800000

# ---- RCCL / AINIC network config ----
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
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# ROCm TE compatibility
export NVTE_FUSED_ATTN_CK=0

# MoE padding flags
export MOE_PADDING=1
export VLLM_FP8_PADDING=1
export VLLM_FP8_ACT_PADDING=1
export VLLM_FP8_WEIGHT_PADDING=1
export VLLM_FP8_REDUCE_CONV=1

# Performance tuning
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1
export HIP_FORCE_DEV_KERNARG=1

# Multi-node RCCL network
export LD_LIBRARY_PATH=/opt/amd-anp/build:/opt/rccl/build/release:/opt/rocm/lib:/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NCCL_NET_PLUGIN_PATH=/opt/amd-anp/build/librccl-net.so

# ---- Paths ----
MODEL_PATH=/shared_nfs/xiaofei/models/Qwen3-235B-A22B
TRAIN_FILE=/shared_nfs/xiaofei/verl/data/gsm8k/train.parquet
TEST_FILE=/shared_nfs/xiaofei/verl/data/gsm8k/test.parquet

# ---- 8-node x 4-GPU parallelism config ----
# Same total GPUs (32), fewer per node to reduce cgroup memory pressure
NNODES=8
GPUS_PER_NODE=4

train_tp=4
train_pp=8
EP=4
ETP=1
CP=1
gen_tp=4  # Reduced from 8 to fit within 4 GPUs/node (single-node sglang)

# DP = total_gpus / (TP * PP) = 32 / 32 = 1
# EP=4 fits within TP*DP = 4*1 = 4 GPUs per PP stage

# ---- Algorithm ----
adv_estimator=grpo
use_kl_loss=True
kl_loss_coef=0.001
clip_ratio_low=0.2
clip_ratio_high=0.28

# ---- Sequence lengths ----
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))

# ---- Batch config ----
train_prompt_bsz=32
n_resp_per_prompt=8
train_prompt_mini_bsz=16

# ---- Offload config ----
offload=True
optimizer_offload_fraction=1.0

# ---- Dynamic batch size ----
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 10 / 10))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))

# ---- Experiment naming ----
project_name='qwen3_235b_megatron'
exp_name="grpo_sglang_8node_4gpu_pp${train_pp}_tp${train_tp}_ep${EP}"
CKPTS_DIR=/shared_nfs/xiaofei/verl/checkpoints/${project_name}/${exp_name}

# Preflight check
python3 - <<PY
tp, pp, ep = $train_tp, $train_pp, $EP
world_size = $NNODES * $GPUS_PER_NODE
dp = world_size // (tp * pp)
print(f"[preflight] nnodes=$NNODES, gpus_per_node=$GPUS_PER_NODE, world={world_size}")
print(f"[preflight] tp={tp}, pp={pp}, ep={ep}, dp={dp}, gen_tp=$gen_tp")
print(f"[preflight] batch={$train_prompt_bsz}, mini_batch={$train_prompt_mini_bsz}, n={$n_resp_per_prompt}")
assert tp * pp <= world_size, f"tp*pp={tp*pp} > world_size={world_size}"
assert ep <= tp * dp, f"ep={ep} > tp*dp={tp*dp}"
assert $gen_tp <= $GPUS_PER_NODE or $gen_tp % $GPUS_PER_NODE == 0, \
    f"gen_tp=$gen_tp must be <= gpus_per_node=$GPUS_PER_NODE or divisible"
print("[preflight] OK — 8-node x 4-GPU config validated")
PY

# Install TMS v0.0.9 from NFS wheel
echo "[pre-flight] Installing torch_memory_saver v0.0.9 from NFS wheel ..."
python3 - <<'INSTALL_TMS'
import ray, os, subprocess, sys
ray.init(address="auto")
@ray.remote(num_cpus=0.001)
def install():
    r = subprocess.run([sys.executable, "-m", "pip", "install",
        "/shared_nfs/xiaofei/wheels/torch_memory_saver-0.0.9-cp310-cp310-linux_x86_64.whl",
        "--no-deps", "--force-reinstall"], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return f"FAIL {os.uname().nodename}: {r.stderr[-200:]}"
    v = subprocess.run([sys.executable, "-c",
        "import torch_memory_saver; print(hasattr(torch_memory_saver, 'torch_memory_saver'))"],
        capture_output=True, text=True, timeout=15)
    return f"OK {os.uname().nodename}: attr={v.stdout.strip()}"
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
for n in ray.nodes():
    if n["Alive"]:
        s = NodeAffinitySchedulingStrategy(node_id=n["NodeID"], soft=False)
        print(f"  [pre-flight] {ray.get(install.options(scheduling_strategy=s).remote())}")
ray.shutdown()
INSTALL_TMS
echo "[pre-flight] torch_memory_saver installation complete"

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.truncation='left' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP} \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend='flash' \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    '+actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache=True' \
    '+actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=False' \
    '+actor_rollout_ref.rollout.engine_kwargs.sglang.enable_memory_saver=True' \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    actor_rollout_ref.nccl_timeout=3600 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.strategy=megatron \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=100 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    2>&1 | tee /shared_nfs/xiaofei/verl/megatron_235b_8node_4gpu.log
