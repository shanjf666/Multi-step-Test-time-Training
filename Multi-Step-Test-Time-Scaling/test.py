# tools/reconstruct_hf_from_fsdp.py
import argparse
import os
import datetime
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.fsdp_utils import get_fsdp_full_state_dict, fsdp_version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def init_dist():
    # init process group only if not initialized
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # use datetime.timedelta instead of non-existent torch.timedelta
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=600))

def build_model_from_config(hf_dir, torch_dtype=torch.float32, trust_remote_code=False):
    config = AutoConfig.from_pretrained(hf_dir, trust_remote_code=trust_remote_code)
    # 根据你的模型类型改这里的类（如果是自定义实现可能需要 trust_remote_code=True）
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
    return model, config

def wrap_with_fsdp(model):
    # 最简单的包装：直接用 FSDP(model)；如果训练时用了复杂 wrap_policy 需要复现
    device = torch.device("cpu")
    if torch.cuda.is_available():
        # 使用 LOCAL_RANK 来决定每个进程对应的 GPU
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    model.to(device)
    fsdp_model = FSDP(model)
    return fsdp_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_actor_dir", required=True, help="Path to actor/ directory containing model_world_size_*_rank_*.pt and fsdp_config.json")
    parser.add_argument("--huggingface_dir", required=True, help="Path to huggingface/ dir (config.json, tokenizer files).")
    parser.add_argument("--output_dir", required=True, help="Where to save reconstructed HF model (rank0).")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    # initialize distributed
    init_dist()

    # after init, we can get rank/world_size
    # if not running under a launcher, these calls might still error; guard them
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    except Exception:
        # single-process fallback
        rank = 0
        world_size = 1

    print(f"Rank {rank}/{world_size} starting...")

    # build HF model from config (weights are random/empty)
    model, config = build_model_from_config(args.huggingface_dir, torch_dtype=torch.float32, trust_remote_code=args.trust_remote_code)

    # wrap with FSDP (must be consistent enough to allow load_state_dict of per-rank shard)
    fsdp_model = wrap_with_fsdp(model)

    # create checkpoint manager and load per-rank shards
    ckpt_manager = FSDPCheckpointManager(model=fsdp_model, optimizer=None, lr_scheduler=None, processing_class=None, checkpoint_config=None)
    ckpt_manager.load_checkpoint(local_path=args.checkpoint_actor_dir, hdfs_path=None, del_local_after_load=False)

    # wait for all ranks (if dist initialized)
    if dist.is_initialized():
        dist.barrier()

    # rank 0 extract full state dict and save HF model
    if rank == 0:
        # get full state dict (offload_to_cpu True avoids GPU OOM)
        full_state = get_fsdp_full_state_dict(fsdp_model, offload_to_cpu=True, rank0_only=True)
        # create HF model instance to load state + save
        hf_model = AutoModelForCausalLM.from_config(config)
        hf_model.load_state_dict(full_state, strict=False)
        tokenizer = AutoTokenizer.from_pretrained(args.huggingface_dir, trust_remote_code=args.trust_remote_code)
        os.makedirs(args.output_dir, exist_ok=True)
        hf_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved reconstructed HF model to {args.output_dir}")

    if dist.is_initialized():
        dist.barrier()
    print(f"Rank {rank} done.")

if __name__ == "__main__":
    main()
