#!/bin/bash
set -e

cd /shared_nfs/xiaofei/verl

# Force reload Python modules from source (clear stale bytecode cache)
echo "[pre-flight] Cleaning __pycache__ under verl/utils/megatron ..."
find /shared_nfs/xiaofei/verl/verl/utils/megatron -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find /shared_nfs/xiaofei/verl/verl/utils/checkpoint -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find /shared_nfs/xiaofei/verl/verl/workers -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "[pre-flight] __pycache__ cleaned"

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# verl uses CUDA_VISIBLE_DEVICES keyword even on ROCm (via HIP compatibility)
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1

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

# Reduce GPU memory fragmentation
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Disable aiter to preserve weight_loader attribute for MoE weight sync
export SGLANG_USE_AITER=0

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

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MODEL_PATH=/shared_nfs/xiaofei/models/Qwen3-30B-A3B

# --- Parallelism config ---
NNODES=2
TRAIN_TP=1
TRAIN_EP=8
TRAIN_PP=1
ROLLOUT_TP=4

# DP = (NNODES * GPUS_PER_NODE) / (TP * EP * PP) = 16 / 8 = 2
DP=$(( NNODES * GPUS_PER_NODE / (TRAIN_TP * TRAIN_EP * TRAIN_PP) ))

# --- Batch config ---
TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=64
ROLLOUT_N=8

# Linear Scaling Rule: lr = base_lr * DP
LR=$(python3 -c "print(5e-7 * $DP)")

# Preflight batch-config check
python3 - <<PY
tp, ep, pp = $TRAIN_TP, $TRAIN_EP, $TRAIN_PP
world_size = $NNODES * $GPUS_PER_NODE
dp = world_size // (tp * ep * pp)
n = $ROLLOUT_N
train_batch_size = $TRAIN_BATCH_SIZE
ppo_mini_batch_size = $PPO_MINI_BATCH_SIZE
if ppo_mini_batch_size > train_batch_size:
    raise SystemExit(
        f"ppo_mini_batch_size={ppo_mini_batch_size} must be <= train_batch_size={train_batch_size}")
if ppo_mini_batch_size < dp or ppo_mini_batch_size % dp != 0:
    raise SystemExit(
        f"ppo_mini_batch_size={ppo_mini_batch_size} must be >= dp={dp} and divisible by dp")
print(f"[preflight] OK: nnodes=$NNODES, world={world_size}, tp={tp}, ep={ep}, "
      f"dp={dp}, n={n}, batch={train_batch_size}, mini_batch={ppo_mini_batch_size}, lr=$LR")
PY

python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files=/shared_nfs/xiaofei/verl/data/gsm8k/train.parquet \
    data.val_files=/shared_nfs/xiaofei/verl/data/gsm8k/test.parquet \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${TRAIN_EP} \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.grad_offload=False \
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend='flash' \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    '+actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache=True' \
    '+actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=False' \
    actor_rollout_ref.nccl_timeout=1200 \
    actor_rollout_ref.rollout.server.timeout=600 \
    actor_rollout_ref.rollout.server.max_attempts=10 \
    actor_rollout_ref.rollout.server.retry_delay=10 \
    actor_rollout_ref.rollout.server.max_start_wait_time=600 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.strategy=megatron \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${TRAIN_EP} \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.attention_backend='flash' \
    actor_rollout_ref.ref.megatron.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.04 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=qwen3_30b_megatron \
    trainer.default_local_dir=/shared_nfs/xiaofei/verl/checkpoints/qwen3_30b_megatron/grpo_sglang_2node_v4 \
    trainer.experiment_name=grpo_sglang_2node_v4 \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    2>&1 | tee /shared_nfs/xiaofei/verl/megatron_30b_2node.log
