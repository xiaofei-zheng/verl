"""Convert Megatron-Core dist_ckpt to HuggingFace safetensors format.

Usage:
    torchrun --nproc_per_node=8 scripts/converter_mcore_to_hf.py \
        --ckpt_dir /path/to/global_step_N/actor \
        --hf_model_path /path/to/original/hf/model \
        --output_dir /path/to/output/hf/model \
        --ep_size 8
"""

import argparse
import gc
import os
import warnings

import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing, parallel_state as mpu
from megatron.core.dist_checkpointing.serialization import StrictHandling
from megatron.core.models.gpt.gpt_model import ModelType
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from verl.models.mcore import hf_to_mcore_config, init_mcore_model
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.megatron_utils import get_model, unwrap_model


def parse_args():
    p = argparse.ArgumentParser(description="Convert Megatron dist_ckpt → HuggingFace")
    p.add_argument("--ckpt_dir", required=True, help="Actor checkpoint dir (contains dist_ckpt/ and huggingface/)")
    p.add_argument("--hf_model_path", required=True, help="Original HF model path (for config & architecture)")
    p.add_argument("--output_dir", required=True, help="Output HF model directory")
    p.add_argument("--ep_size", type=int, default=8)
    p.add_argument("--tp_size", type=int, default=1)
    p.add_argument("--pp_size", type=int, default=1)
    return p.parse_args()


@torch.inference_mode()
def extract_hf_state_dict(mcore_model, hf_config, ep_rank, ep_size):
    """Extract HF-format state dict from a single EP rank's Megatron model."""
    model = mcore_model
    sd = {}

    sd["model.embed_tokens.weight"] = model.embedding.word_embeddings.weight.data.cpu()

    num_attention_heads = hf_config.num_attention_heads
    num_key_value_heads = hf_config.num_key_value_heads
    hidden_size = hf_config.hidden_size
    head_dim = getattr(hf_config, "head_dim", hidden_size // num_attention_heads)
    num_query_groups = num_key_value_heads
    q_per_kv = num_attention_heads // num_key_value_heads

    for layer_idx, layer in enumerate(model.decoder.layers):
        prefix = f"model.layers.{layer_idx}"
        sd[f"{prefix}.input_layernorm.weight"] = layer.self_attention.linear_qkv.layer_norm_weight.data.cpu()

        qkv_w = layer.self_attention.linear_qkv.weight.data.cpu()
        qkv_w = qkv_w.view(num_query_groups, q_per_kv + 2, head_dim, hidden_size)
        q_w = qkv_w[:, :q_per_kv, :, :].reshape(-1, hidden_size)
        k_w = qkv_w[:, q_per_kv, :, :].reshape(-1, hidden_size)
        v_w = qkv_w[:, q_per_kv + 1, :, :].reshape(-1, hidden_size)
        sd[f"{prefix}.self_attn.q_proj.weight"] = q_w
        sd[f"{prefix}.self_attn.k_proj.weight"] = k_w
        sd[f"{prefix}.self_attn.v_proj.weight"] = v_w

        has_bias = (
            hasattr(layer.self_attention.linear_qkv, "bias")
            and layer.self_attention.linear_qkv.bias is not None
            and layer.self_attention.linear_qkv.bias.numel() > 0
        )
        if has_bias:
            qkv_b = layer.self_attention.linear_qkv.bias.data.cpu()
            qkv_b = qkv_b.view(num_query_groups, q_per_kv + 2, head_dim)
            sd[f"{prefix}.self_attn.q_proj.bias"] = qkv_b[:, :q_per_kv, :].reshape(-1)
            sd[f"{prefix}.self_attn.k_proj.bias"] = qkv_b[:, q_per_kv, :].reshape(-1)
            sd[f"{prefix}.self_attn.v_proj.bias"] = qkv_b[:, q_per_kv + 1, :].reshape(-1)

        if hasattr(layer.self_attention, "q_layernorm") and layer.self_attention.q_layernorm is not None:
            sd[f"{prefix}.self_attn.q_norm.weight"] = layer.self_attention.q_layernorm.weight.data.cpu()
        if hasattr(layer.self_attention, "k_layernorm") and layer.self_attention.k_layernorm is not None:
            sd[f"{prefix}.self_attn.k_norm.weight"] = layer.self_attention.k_layernorm.weight.data.cpu()

        sd[f"{prefix}.self_attn.o_proj.weight"] = layer.self_attention.linear_proj.weight.data.cpu()

        if hasattr(layer.mlp, "router"):
            sd[f"{prefix}.mlp.gate.weight"] = layer.mlp.router.weight.data.cpu()
            if hasattr(layer.mlp.router, "expert_bias") and layer.mlp.router.expert_bias is not None:
                sd[f"{prefix}.mlp.gate.e_score_correction_bias"] = layer.mlp.router.expert_bias.data.cpu()
            sd[f"{prefix}.post_attention_layernorm.weight"] = layer.pre_mlp_layernorm.weight.data.cpu()

            num_experts = hf_config.num_experts
            num_local_experts = num_experts // ep_size
            expert_start = ep_rank * num_local_experts

            if hasattr(layer.mlp.experts, "linear_fc1"):
                for local_i in range(num_local_experts):
                    global_i = expert_start + local_i
                    fc1_w = getattr(layer.mlp.experts.linear_fc1, f"weight{local_i}").data.cpu()
                    gate_w, up_w = fc1_w.chunk(2, dim=0)
                    sd[f"{prefix}.mlp.experts.{global_i}.gate_proj.weight"] = gate_w
                    sd[f"{prefix}.mlp.experts.{global_i}.up_proj.weight"] = up_w
                    fc2_w = getattr(layer.mlp.experts.linear_fc2, f"weight{local_i}").data.cpu()
                    sd[f"{prefix}.mlp.experts.{global_i}.down_proj.weight"] = fc2_w
            else:
                for local_i, expert in enumerate(layer.mlp.experts.local_experts):
                    global_i = expert_start + local_i
                    fc1_w = expert.linear_fc1.weight.data.cpu()
                    gate_w, up_w = fc1_w.chunk(2, dim=0)
                    sd[f"{prefix}.mlp.experts.{global_i}.gate_proj.weight"] = gate_w
                    sd[f"{prefix}.mlp.experts.{global_i}.up_proj.weight"] = up_w
                    sd[f"{prefix}.mlp.experts.{global_i}.down_proj.weight"] = expert.linear_fc2.weight.data.cpu()

            if hasattr(layer.mlp, "shared_experts") and layer.mlp.shared_experts is not None:
                shared_fc1 = layer.mlp.shared_experts.linear_fc1.weight.data.cpu()
                gate_w, up_w = shared_fc1.chunk(2, dim=0)
                sd[f"{prefix}.mlp.shared_experts.gate_proj.weight"] = gate_w
                sd[f"{prefix}.mlp.shared_experts.up_proj.weight"] = up_w
                sd[f"{prefix}.mlp.shared_experts.down_proj.weight"] = (
                    layer.mlp.shared_experts.linear_fc2.weight.data.cpu()
                )
        else:
            sd[f"{prefix}.post_attention_layernorm.weight"] = layer.mlp.linear_fc1.layer_norm_weight.data.cpu()
            fc1_w = layer.mlp.linear_fc1.weight.data.cpu()
            gate_w, up_w = fc1_w.chunk(2, dim=0)
            sd[f"{prefix}.mlp.gate_proj.weight"] = gate_w
            sd[f"{prefix}.mlp.up_proj.weight"] = up_w
            sd[f"{prefix}.mlp.down_proj.weight"] = layer.mlp.linear_fc2.weight.data.cpu()

    sd["model.norm.weight"] = model.decoder.final_layernorm.weight.data.cpu()
    if not getattr(hf_config, "tie_word_embeddings", False):
        sd["lm_head.weight"] = model.output_layer.weight.data.cpu()

    return sd


def main():
    args = parse_args()

    if "WORLD_SIZE" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = str(args.ep_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29510"

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    get_torch_device().set_device(f"{get_device_name()}:{local_rank}")

    assert world_size == args.tp_size * args.ep_size * args.pp_size, (
        f"world_size={world_size} != tp*ep*pp={args.tp_size}*{args.ep_size}*{args.pp_size}"
    )

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        expert_model_parallel_size=args.ep_size,
    )
    model_parallel_cuda_manual_seed(0)

    hf_config = AutoConfig.from_pretrained(args.hf_model_path)
    if rank == 0:
        print(f"HF config: {hf_config.architectures}", flush=True)

    tfconfig = hf_to_mcore_config(hf_config, torch.bfloat16)
    tfconfig.use_cpu_initialization = True
    tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)

    def model_provider(pre_process, post_process):
        return init_mcore_model(
            tfconfig, hf_config, pre_process, post_process,
            share_embeddings_and_output_weights=tie_word_embeddings, value=False,
        )

    if rank == 0:
        print("Creating Megatron model (CPU)...", flush=True)
    from accelerate import init_empty_weights
    with init_empty_weights():
        model = get_model(
            model_provider_func=model_provider,
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=False,
            transformer_config=tfconfig,
        )
    model[0].module = model[0].module.to_empty(device="cpu")

    dist_ckpt_path = os.path.join(args.ckpt_dir, "dist_ckpt")
    if rank == 0:
        print(f"Loading dist_ckpt from {dist_ckpt_path}...", flush=True)

    ssd = unwrap_model(model[0]).sharded_state_dict()
    dist_checkpointing.load(ssd, dist_ckpt_path, strict=StrictHandling.ASSUME_OK_UNEXPECTED)

    if rank == 0:
        print("Dist checkpoint loaded. Extracting HF state dict...", flush=True)

    ep_rank = mpu.get_expert_model_parallel_rank()
    local_sd = extract_hf_state_dict(unwrap_model(model[0]), hf_config, ep_rank, args.ep_size)

    if rank == 0:
        print(f"Rank 0 extracted {len(local_sd)} keys. Gathering expert weights...", flush=True)

    if args.ep_size > 1:
        import tempfile
        gather_dir = os.path.join(args.output_dir, "_expert_gather")
        os.makedirs(gather_dir, exist_ok=True)

        my_expert_sd = {k: v for k, v in local_sd.items() if ".mlp.experts." in k}
        shard_path = os.path.join(gather_dir, f"rank_{rank}.pt")
        torch.save(my_expert_sd, shard_path)
        if rank == 0:
            print(f"Each rank saved expert weights to {gather_dir}/", flush=True)

        dist.barrier()

        if rank == 0:
            for src_rank in range(1, world_size):
                src_path = os.path.join(gather_dir, f"rank_{src_rank}.pt")
                src_sd = torch.load(src_path, map_location="cpu", weights_only=True)
                local_sd.update(src_sd)
                del src_sd
                print(f"  Loaded expert weights from rank {src_rank}", flush=True)

        dist.barrier()
        if rank == 0:
            import shutil
            shutil.rmtree(gather_dir, ignore_errors=True)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Total keys: {len(local_sd)}. Saving to {args.output_dir}...", flush=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from accelerate import init_empty_weights as iew
            with iew():
                hf_model = AutoModelForCausalLM.from_pretrained(args.hf_model_path, torch_dtype=torch.bfloat16)

        hf_model.save_pretrained(args.output_dir, state_dict=local_sd, max_shard_size="5GB")

        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path)
        tokenizer.save_pretrained(args.output_dir)

        print(f"Done! HuggingFace model saved to {args.output_dir}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
