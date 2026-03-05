#!/bin/bash
set -e

cd /shared_nfs/xiaofei/verl

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Ray < 2.45.0 uses ROCR, Ray >= 2.45.0 uses HIP
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export SGLANG_DISABLE_FA3=1

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
SAVE_DIR=${SAVE_DIR:-/shared_nfs/xiaofei/verl_checkpoints_fsdp1_multinode}

# Linear Scaling Rule: lr = base_lr * NNODES
LR=$(python3 -c "print(1e-6 * $NNODES)")

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

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
    actor_rollout_ref.actor.strategy=fsdp \
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
    trainer.experiment_name=qwen3_8b_grpo_fsdp1_${NNODES}node \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=100 \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.test_freq=5 \
    trainer.total_epochs=2
