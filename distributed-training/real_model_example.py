#!/usr/bin/env python3
"""
Real Model Example: Fine-tuning GPT-2 with Different Distributed Strategies
----------------------------------------------------------------------------

Configuration is done directly in the CONFIG dictionary below (no command-line args).
Edit the CONFIG values to change model, strategy, batch size, etc.

(Updated: fixes for dataset formatting, device handling, and DeepSpeed/DDP usage)
"""
import os
import time
import random
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
import deepspeed


# ========================================================================
# Configuration
# ========================================================================

MODEL_CONFIGS = {
    "gpt2": {"name": "gpt2", "params": "124M", "hidden_size": 768, "layers": 12},
    "gpt2-medium": {"name": "gpt2-medium", "params": "355M", "hidden_size": 1024, "layers": 24},
    "gpt2-large": {"name": "gpt2-large", "params": "774M", "hidden_size": 1280, "layers": 36},
    "gpt2-xl": {"name": "gpt2-xl", "params": "1.5B", "hidden_size": 1600, "layers": 48},
}


# ========================================================================
# Utilities
# ========================================================================

def setup_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """Initialize distributed training."""
    # Local launcher (torchrun / deepspeed) sets these env vars
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        # init process group (NCCL is preferred for GPUs)
        dist.init_process_group(backend="nccl", init_method="env://")

    # Only set CUDA device if available
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return rank, local_rank, world_size, device


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def print_memory_stats(device_or_index, rank, prefix=""):
    """Print GPU memory usage. Accepts torch.device or int GPU index."""
    if rank != 0:
        return
    if not torch.cuda.is_available():
        print(f"{prefix}CUDA not available - skipping memory stats")
        return

    if isinstance(device_or_index, torch.device):
        dev = device_or_index
    else:
        # assume int index
        dev = torch.device(f"cuda:{int(device_or_index)}")

    allocated = torch.cuda.memory_allocated(dev) / 1e9
    reserved = torch.cuda.memory_reserved(dev) / 1e9
    max_allocated = torch.cuda.max_memory_allocated(dev) / 1e9
    print(f"{prefix}GPU Memory: {allocated:.2f} GB allocated | "
          f"{reserved:.2f} GB reserved | {max_allocated:.2f} GB peak")


def count_parameters(model):
    """Count trainable parameters (works on raw model or DDP-wrapped model)."""
    # If wrapped in DDP, access .module
    m = model.module if hasattr(model, "module") else model
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ========================================================================
# Dataset Preparation
# ========================================================================

def prepare_dataset(tokenizer, max_length=512, num_samples=1000, split="train"):
    """
    Prepare a text dataset for training using WikiText-2.
    This returns a `datasets.Dataset` with columns ['input_ids', 'attention_mask'] in torch format.
    """
    print("📦 Loading dataset (WikiText-2)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    def tokenize_batch(examples):
        # return lists of ints (not tensors) so `datasets` can set_format later
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True
        )
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
    )

    if len(tokenized) > num_samples:
        tokenized = tokenized.select(range(num_samples))

    # set format so DataLoader yields PyTorch tensors directly
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return tokenized


# ========================================================================
# Training Functions
# ========================================================================

def train_epoch_ddp(model, dataloader, optimizer, scheduler, device, rank, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0 and rank == 0:
            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f} | LR: {lr:.2e}")

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, num_batches)

    if rank == 0:
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Avg Loss: {avg_loss:.4f}")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Throughput: {len(dataloader.dataset) / max(1e-9, elapsed):.2f} samples/sec")

    return avg_loss


def train_epoch_deepspeed(model_engine, dataloader, rank, epoch):
    model_engine.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(model_engine.device)
        attention_mask = batch["attention_mask"].to(model_engine.device)

        outputs = model_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        # outputs.loss is expected when using Transformers models with LM head
        loss = outputs.loss

        model_engine.backward(loss)
        model_engine.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0 and rank == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, num_batches)

    if rank == 0:
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Avg Loss: {avg_loss:.4f}")
        print(f"   Time: {elapsed:.2f}s")

    return avg_loss


# ========================================================================
# Configuration Settings (Edit these directly instead of using command-line args)
# ========================================================================

CONFIG = {
    "model": "gpt2-medium",  # Options: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
    "strategy": "zero2",     # Options: "dp", "zero1", "zero2", "zero3", "offload", "infinity"
    "batch_size": 4,         # Batch size per GPU
    "epochs": 3,             # Number of training epochs
    "max_length": 512,       # Maximum sequence length
    "num_samples": 1000,     # Number of training samples to use
    "seed": 42,              # Random seed for reproducibility
    "deepspeed_config": None # DeepSpeed config file (auto-selected if None)
}


# ========================================================================
# Main Training Loop
# ========================================================================

def main():
    # Use hardcoded configuration
    class Args:
        def __init__(self, config):
            self.model = config["model"]
            self.strategy = config["strategy"]
            self.batch_size = config["batch_size"]
            self.epochs = config["epochs"]
            self.max_length = config["max_length"]
            self.num_samples = config["num_samples"]
            self.seed = config["seed"]
            self.deepspeed = config["deepspeed_config"]
            self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    args = Args(CONFIG)

    setup_seed(args.seed)
    rank, local_rank, world_size, device = setup_distributed()

    model_info = MODEL_CONFIGS[args.model]

    if rank == 0:
        print("=" * 80)
        print(f"🚀 Fine-tuning {args.model} ({model_info['params']} parameters)")
        print(f"📋 Strategy: {args.strategy.upper()}")
        print("=" * 80)
        print(f"World size: {world_size} GPUs")
        print(f"Batch size: {args.batch_size} per GPU")
        print(f"Epochs: {args.epochs}")
        print(f"Samples: {args.num_samples}")
        print("=" * 80)

    # Load tokenizer & model (on CPU first to avoid unnecessary GPU memory usage)
    if rank == 0:
        print("\n📥 Loading model and tokenizer...")

    tokenizer = GPT2Tokenizer.from_pretrained(model_info["name"])
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_info["name"])

    if rank == 0:
        num_params = count_parameters(model)
        print(f"✅ Model loaded: {num_params:,} parameters ({num_params/1e6:.1f}M)")
        param_memory = num_params * 2 / 1e9  # FP16 rough
        print(f"\n💾 Estimated Memory (FP16 rough):")
        print(f"   Parameters: {param_memory:.2f} GB")
        print(f"   Gradients: {param_memory:.2f} GB")
        print(f"   Optimizer: {param_memory * 6:.2f} GB")
        print(f"   Total: {param_memory * 9:.2f} GB per GPU (without ZeRO)")

    # Prepare dataset
    if rank == 0:
        print(f"\n📊 Preparing dataset...")

    dataset = prepare_dataset(tokenizer, args.max_length, args.num_samples)
    if rank == 0:
        print(f"✅ Dataset ready: {len(dataset)} samples")

    # Strategy-specific setup
    if args.strategy == "dp":
        # Data Parallelism with DDP
        model = model.to(device)

        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        total_steps = len(dataloader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

        if rank == 0:
            print("\n✅ Data Parallelism setup complete")
            print_memory_stats(device, rank, prefix="   ")

        if rank == 0:
            print("\n" + "=" * 80)
            print("🚂 Starting Training (Data Parallelism)")
            print("=" * 80)

        for epoch in range(1, args.epochs + 1):
            if sampler:
                sampler.set_epoch(epoch)

            # reset peak
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            train_epoch_ddp(model, dataloader, optimizer, scheduler, device, rank, epoch)

            if rank == 0:
                print_memory_stats(device, rank, prefix="   ")
                print()

    else:
        # DeepSpeed strategies
        if args.deepspeed is None:
            config_map = {
                "zero1": "ds_config_stage1.json",
                "zero2": "ds_config_stage2.json",
                "zero3": "ds_config_stage3.json",
                "offload": "ds_config_offload.json",
                "infinity": "ds_config_infinity.json",
            }
            args.deepspeed = config_map[args.strategy]

        if rank == 0:
            print(f"\n⚙️  Initializing DeepSpeed with {args.deepspeed}...")

        # DeepSpeed expects training_data or a dataloader; it can create its own dataloader from a Dataset
        model_engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=dataset,
            config=args.deepspeed,
        )

        if rank == 0:
            print(f"✅ DeepSpeed initialized ({args.strategy.upper()})")
            # model_engine.device is the device used by DeepSpeed
            print_memory_stats(model_engine.device, rank, prefix="   ")

        if rank == 0:
            print("\n" + "=" * 80)
            print(f"🚂 Starting Training ({args.strategy.upper()})")
            print("=" * 80)

        for epoch in range(1, args.epochs + 1):
            # DeepSpeed dataloader is supplied by deepspeed.initialize
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(model_engine.device)
            train_epoch_deepspeed(model_engine, training_dataloader, rank, epoch)

            if rank == 0:
                print_memory_stats(model_engine.device, rank, prefix="   ")
                print()

    # Final summary
    if rank == 0:
        print("=" * 80)
        print(f"✅ Training Complete!")
        print("=" * 80)
        print(f"\n💡 Strategy: {args.strategy.upper()}")
        print(f"📊 Model: {args.model} ({model_info['params']})")
        print("=" * 80)

    cleanup_distributed()


if __name__ == "__main__":
    main()
