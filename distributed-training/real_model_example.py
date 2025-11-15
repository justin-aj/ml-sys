#!/usr/bin/env python3
"""
===============================================================================
Real Model Example: Fine-tuning GPT-2 with Different Distributed Strategies
===============================================================================

LEARNING-FOCUSED CODE - This code is written for clarity and education, not performance.

PURPOSE:
    Learn how to train large language models using different distributed strategies:
    - Data Parallelism (DP)
    - ZeRO Stage 1, 2, 3
    - ZeRO-Offload (CPU memory)
    - ZeRO-Infinity (NVMe storage)

HOW TO USE:
    1. Edit the CONFIG dictionary below (around line 100)
    2. Run with: python real_model_example.py
    3. For multi-GPU: torchrun --nproc_per_node=4 real_model_example.py
    4. For DeepSpeed: deepspeed --num_gpus=4 real_model_example.py

WHAT YOU'LL LEARN:
    - How to load and prepare a real dataset (WikiText-2)
    - How to load a pre-trained model (GPT-2)
    - How distributed training works (DDP vs DeepSpeed)
    - How ZeRO reduces memory usage
    - How to monitor GPU memory and training progress

===============================================================================
QUICK REFERENCE: CONFIG OPTIONS
===============================================================================

┌─────────────────────────────────────────────────────────────────────────┐
│ PARAMETER       │ OPTIONS                     │ RECOMMENDED             │
├─────────────────────────────────────────────────────────────────────────┤
│ model           │ gpt2                        │ For testing/learning    │
│                 │ gpt2-medium                 │ For experiments         │
│                 │ gpt2-large                  │ For ZeRO demonstration  │
│                 │ gpt2-xl                     │ For advanced ZeRO       │
├─────────────────────────────────────────────────────────────────────────┤
│ strategy        │ dp                          │ Baseline (high memory)  │
│                 │ zero1                       │ 4× memory reduction     │
│                 │ zero2                       │ 8× reduction (best!)    │
│                 │ zero3                       │ N× reduction (largest)  │
│                 │ offload                     │ Use CPU RAM             │
│                 │ infinity                    │ Use NVMe disk           │
├─────────────────────────────────────────────────────────────────────────┤
│ batch_size      │ 1-32 (powers of 2)          │ 4 for medium models     │
│                 │                             │ Reduce if OOM error     │
├─────────────────────────────────────────────────────────────────────────┤
│ epochs          │ 1-100                       │ 3 for quick experiments │
│                 │                             │ 10+ for real training   │
├─────────────────────────────────────────────────────────────────────────┤
│ max_length      │ 128, 256, 512, 1024         │ 512 (good balance)      │
│                 │                             │ Lower = less memory     │
├─────────────────────────────────────────────────────────────────────────┤
│ num_samples     │ 100-36000 or -1 (all)       │ 1000 for testing        │
│                 │                             │ -1 for full training    │
├─────────────────────────────────────────────────────────────────────────┤
│ seed            │ Any integer                 │ 42 (standard)           │
├─────────────────────────────────────────────────────────────────────────┤
│ deepspeed_config│ None (auto) or file path    │ None (recommended)      │
└─────────────────────────────────────────────────────────────────────────┘

MEMORY USAGE COMPARISON (gpt2-medium on 4 GPUs):
    Strategy    │ Memory per GPU │ Speed      │ Use When
    ────────────┼────────────────┼────────────┼───────────────────────
    dp          │ 28 GB          │ 100%       │ Model fits easily
    zero1       │ 22 GB          │ 100%       │ Optimizer is bottleneck
    zero2       │ 18 GB          │ 95%        │ Most use cases ⭐
    zero3       │ 10 GB          │ 85%        │ Large models
    offload     │ 15 GB + CPU    │ 70%        │ Limited GPU memory
    infinity    │ 8 GB + NVMe    │ 50%        │ Huge models (10B+)

EXAMPLE COMMANDS:
    # Test with small model (Data Parallelism)
    torchrun --nproc_per_node=4 real_model_example.py
    
    # Train with ZeRO-2 (recommended)
    deepspeed --num_gpus=4 real_model_example.py
    
    # Large model with ZeRO-3
    deepspeed --num_gpus=4 real_model_example.py
    (Set strategy="zero3" and model="gpt2-large" in CONFIG)

===============================================================================
"""

# ========================================================================
# Step 1: Import Required Libraries
# ========================================================================

import os          # For environment variables and file operations
import time        # For measuring training time
import random      # For setting random seeds
import torch       # PyTorch - main deep learning framework
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# HuggingFace Transformers - provides pre-trained models
from transformers import (
    GPT2LMHeadModel,                    # The GPT-2 model
    GPT2Tokenizer,                      # Converts text to numbers
    get_linear_schedule_with_warmup     # Learning rate scheduler
)

# HuggingFace Datasets - provides easy access to datasets
from datasets import load_dataset

# DeepSpeed - Microsoft's library for efficient distributed training
import deepspeed


# ========================================================================
# Step 2: Model Configuration
# ========================================================================

# Available GPT-2 models from HuggingFace
# Each model has different size and memory requirements
MODEL_CONFIGS = {
    "gpt2": {
        "name": "gpt2", 
        "params": "124M",      # 124 million parameters
        "hidden_size": 768,    # Size of hidden layers
        "layers": 12           # Number of transformer layers
    },
    "gpt2-medium": {
        "name": "gpt2-medium",
        "params": "355M",      # 355 million parameters
        "hidden_size": 1024,
        "layers": 24
    },
    "gpt2-large": {
        "name": "gpt2-large",
        "params": "774M",      # 774 million parameters
        "hidden_size": 1280,
        "layers": 36
    },
    "gpt2-xl": {
        "name": "gpt2-xl",
        "params": "1.5B",      # 1.5 billion parameters
        "hidden_size": 1600,
        "layers": 48
    },
}


# ========================================================================
# Step 3: Training Configuration (EDIT THESE VALUES!)
# ========================================================================

# NOTE: The actual CONFIG dictionary is defined later in the file (around line 680)
# after all the helper functions. This keeps the educational flow better.
# Jump to "Step 7: Main Training Loop" section to edit CONFIG.


# ========================================================================
# Step 4: Utility Functions
# ========================================================================

def setup_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Why? Machine learning uses randomness (weight initialization, data shuffling).
    Setting seeds ensures you get the same results each time you run.
    """
    random.seed(seed)              # Python's random module
    torch.manual_seed(seed)        # PyTorch CPU operations
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU operations


def setup_distributed():
    """
    Initialize distributed training across multiple GPUs/nodes.
    
    WHAT IS DISTRIBUTED TRAINING?
    When you have multiple GPUs, you want them all to work together.
    This function sets up the communication between GPUs.
    
    ENVIRONMENT VARIABLES:
    When you run with torchrun or deepspeed, they set these variables:
    - RANK: Global rank (0, 1, 2, ... across all GPUs on all machines)
    - LOCAL_RANK: Local rank (0, 1, 2, ... on this machine only)
    - WORLD_SIZE: Total number of GPUs across all machines
    
    EXAMPLE:
    If you have 2 machines with 4 GPUs each (8 GPUs total):
    - Machine 1, GPU 0: RANK=0, LOCAL_RANK=0, WORLD_SIZE=8
    - Machine 1, GPU 1: RANK=1, LOCAL_RANK=1, WORLD_SIZE=8
    - Machine 2, GPU 0: RANK=4, LOCAL_RANK=0, WORLD_SIZE=8
    """
    # Get distributed training info from environment variables
    rank = int(os.environ.get("RANK", 0))           # Which GPU am I globally?
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Which GPU am I locally?
    world_size = int(os.environ.get("WORLD_SIZE", 1))  # How many GPUs total?

    # Only initialize if we have multiple GPUs
    if world_size > 1:
        # NCCL is NVIDIA's library for GPU communication (fastest for GPUs)
        dist.init_process_group(backend="nccl", init_method="env://")

    # Set which GPU this process should use
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")  # Fallback to CPU if no GPU

    return rank, local_rank, world_size, device


def cleanup_distributed():
    """
    Clean up distributed training when done.
    
    This tells PyTorch we're done with multi-GPU communication.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def print_memory_stats(device_or_index, rank, prefix=""):
    """
    Print GPU memory usage.
    
    WHY MONITOR MEMORY?
    Large models can use 10-80+ GB of GPU memory. If you run out,
    training crashes with "CUDA out of memory" error.
    
    MEMORY TYPES:
    - Allocated: Memory actually in use right now
    - Reserved: Memory PyTorch has reserved (but might not be using)
    - Peak: Maximum memory used so far
    
    Args:
        device_or_index: GPU device or index number
        rank: Which GPU (only print for rank 0 to avoid spam)
        prefix: Text to print before memory stats
    """
    # Only print from one GPU to avoid duplicate output
    if rank != 0:
        return
    
    if not torch.cuda.is_available():
        print(f"{prefix}CUDA not available - running on CPU")
        return

    # Handle both torch.device and int index
    if isinstance(device_or_index, torch.device):
        dev = device_or_index
    else:
        dev = torch.device(f"cuda:{int(device_or_index)}")

    # Get memory stats in GB (easier to read than bytes)
    allocated = torch.cuda.memory_allocated(dev) / 1e9
    reserved = torch.cuda.memory_reserved(dev) / 1e9
    max_allocated = torch.cuda.max_memory_allocated(dev) / 1e9
    
    print(f"{prefix}GPU Memory: {allocated:.2f} GB allocated | "
          f"{reserved:.2f} GB reserved | {max_allocated:.2f} GB peak")


def count_parameters(model):
    """
    Count how many parameters (weights) the model has.
    
    WHY COUNT PARAMETERS?
    - Larger models = more memory needed
    - A 1B parameter model in FP16 needs ~2GB just for weights
    - Total memory = weights + gradients + optimizer states
    
    Args:
        model: The neural network (can be wrapped in DDP or not)
    
    Returns:
        Number of trainable parameters
    """
    # If model is wrapped in DDP, unwrap it first
    actual_model = model.module if hasattr(model, "module") else model
    
    # Count all parameters that require gradients (trainable parameters)
    total_params = sum(
        param.numel()  # numel() = number of elements in this parameter
        for param in actual_model.parameters() 
        if param.requires_grad
    )
    
    return total_params


# ========================================================================
# Step 5: Dataset Preparation
# ========================================================================

def prepare_dataset(tokenizer, max_length=512, num_samples=1000, split="train"):
    """
    Load and prepare WikiText-2 dataset for training.
    
    WHAT IS WIKITEXT-2?
    A collection of Wikipedia articles (about 2 million words).
    It's commonly used for language modeling research.
    
    WHAT IS TOKENIZATION?
    Neural networks can't read text directly. Tokenization converts text
    into numbers that the model can understand.
    
    Example: "Hello world" might become [31373, 995]
    
    Args:
        tokenizer: The tokenizer (converts text ↔ numbers)
        max_length: Maximum sequence length in tokens
        num_samples: How many training examples to use
        split: Which split to use ("train", "validation", or "test")
    
    Returns:
        A dataset ready for training (with input_ids and attention_mask)
    """
    print("📦 Loading WikiText-2 dataset from HuggingFace...")
    
    # Load the dataset (automatically downloads if not cached)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    
    def tokenize_batch(examples):
        """
        Convert a batch of text into token IDs.
        
        This function is called on batches of data to speed things up.
        """
        # Tokenize the text
        tokenized = tokenizer(
            examples["text"],
            truncation=True,         # Cut off text longer than max_length
            max_length=max_length,   # Maximum sequence length
            padding="max_length",    # Pad shorter sequences to max_length
            return_attention_mask=True  # Return mask showing which tokens are padding
        )
        return {
            "input_ids": tokenized["input_ids"],           # The token IDs
            "attention_mask": tokenized["attention_mask"]  # 1 for real tokens, 0 for padding
        }

    # Apply tokenization to the entire dataset
    print("🔄 Tokenizing text (converting words → numbers)...")
    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,  # Process multiple examples at once (faster)
        remove_columns=dataset.column_names,  # Remove original text column
    )

    # Only use a subset of data if requested (useful for quick testing)
    if len(tokenized_dataset) > num_samples:
        tokenized_dataset = tokenized_dataset.select(range(num_samples))
        print(f"✂️  Using {num_samples} samples (out of {len(dataset)} available)")

    # Convert to PyTorch tensors so DataLoader can use them
    tokenized_dataset.set_format(
        type="torch", 
        columns=["input_ids", "attention_mask"]
    )

    print(f"✅ Dataset ready: {len(tokenized_dataset)} tokenized samples")
    return tokenized_dataset


# ========================================================================
# Step 6: Training Functions
# ========================================================================

def train_epoch_ddp(model, dataloader, optimizer, scheduler, device, rank, epoch):
    """
    Train for one epoch using Data Parallelism (DDP).
    
    WHAT IS AN EPOCH?
    One complete pass through the entire dataset. If you have 1000 samples
    and batch_size=10, one epoch = 100 batches.
    
    HOW DOES DATA PARALLELISM WORK?
    1. Each GPU gets a different batch of data
    2. Each GPU computes gradients independently
    3. Gradients are averaged across all GPUs (all-reduce)
    4. All GPUs update their model with the same averaged gradients
    
    MEMORY USAGE (DATA PARALLELISM):
    - Each GPU holds a complete copy of the model
    - Each GPU holds a complete copy of optimizer states
    - Very memory inefficient for large models!
    
    Args:
        model: The neural network (wrapped in DDP)
        dataloader: Provides batches of data
        optimizer: Updates model weights (e.g., AdamW)
        scheduler: Adjusts learning rate during training
        device: Which GPU to use
        rank: Which GPU this is (0, 1, 2, ...)
        epoch: Current epoch number
    """
    model.train()  # Set model to training mode (enables dropout, etc.)
    
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    # Loop through all batches in the dataset
    for batch_idx, batch in enumerate(dataloader):
        # Move data to GPU
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Forward pass: compute predictions and loss
        # For language modeling, the task is to predict the next token
        # So we use input_ids both as input AND as labels
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids  # Predict the input itself (language modeling)
        )
        loss = outputs.loss  # Cross-entropy loss (how wrong were predictions?)

        # Backward pass: compute gradients
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients
        
        # Gradient clipping: prevent exploding gradients
        # (Limits maximum gradient value to avoid instability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update model weights
        optimizer.step()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Track statistics
        total_loss += loss.item()
        num_batches += 1

        # Print progress every 10 batches (only from GPU 0)
        if batch_idx % 10 == 0 and rank == 0:
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | LR: {current_lr:.2e}")

    # Epoch finished - print summary
    elapsed_time = time.time() - start_time
    avg_loss = total_loss / max(1, num_batches)

    if rank == 0:
        print(f"\n📊 Epoch {epoch} Summary (Data Parallelism):")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Time: {elapsed_time:.2f} seconds")
        samples_per_sec = len(dataloader.dataset) / max(1e-9, elapsed_time)
        print(f"   Throughput: {samples_per_sec:.2f} samples/second")

    return avg_loss


def train_epoch_deepspeed(model_engine, dataloader, rank, epoch):
    """
    Train for one epoch using DeepSpeed (ZeRO optimization).
    
    WHAT IS DeepSpeed?
    Microsoft's library for efficient distributed training. It implements
    ZeRO (Zero Redundancy Optimizer) which dramatically reduces memory usage.
    
    HOW DOES ZeRO WORK?
    Instead of each GPU holding everything, ZeRO shards (splits) memory:
    
    ZeRO Stage 1: Shard optimizer states only
    - 4× less memory than Data Parallelism
    - Each GPU holds 1/N of optimizer states
    
    ZeRO Stage 2: Shard optimizer states + gradients
    - 8× less memory than Data Parallelism
    - Each GPU holds 1/N of optimizer AND gradients
    
    ZeRO Stage 3: Shard everything (params + grads + optimizer)
    - N× less memory (N = number of GPUs)
    - Each GPU only holds 1/N of the model!
    - Needs communication to gather parameters during forward/backward
    
    EXAMPLE:
    If you have 4 GPUs with ZeRO-3:
    - GPU 0 holds layers 0-2
    - GPU 1 holds layers 3-5
    - GPU 2 holds layers 6-8
    - GPU 3 holds layers 9-11
    When GPU 0 needs layer 5, it asks GPU 1 for it (all-gather operation)
    
    Args:
        model_engine: DeepSpeed engine (wraps model + optimizer)
        dataloader: Provides batches of data
        rank: Which GPU this is
        epoch: Current epoch number
    """
    model_engine.train()  # Set to training mode
    
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    # Loop through all batches
    for batch_idx, batch in enumerate(dataloader):
        # Move data to the device DeepSpeed is using
        input_ids = batch["input_ids"].to(model_engine.device)
        attention_mask = batch["attention_mask"].to(model_engine.device)

        # Forward pass
        outputs = model_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        loss = outputs.loss

        # Backward pass (DeepSpeed handles gradient sharding automatically!)
        model_engine.backward(loss)
        
        # Optimizer step (DeepSpeed handles parameter updates and communication)
        model_engine.step()

        # Track statistics
        total_loss += loss.item()
        num_batches += 1

        # Print progress every 10 batches
        if batch_idx % 10 == 0 and rank == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f}")

    # Epoch finished - print summary
    elapsed_time = time.time() - start_time
    avg_loss = total_loss / max(1, num_batches)

    if rank == 0:
        print(f"\n📊 Epoch {epoch} Summary (DeepSpeed/ZeRO):")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Time: {elapsed_time:.2f} seconds")

    return avg_loss


# ========================================================================
# Configuration Settings (Edit these directly instead of using command-line args)
# ========================================================================

CONFIG = {
    "model": "gpt2-medium",
    "strategy": "dp",           # Data Parallelism
    "batch_size": 4,
    "epochs": 2,
    "max_length": 512,
    "num_samples": 1000,
    "seed": 42,
    "deepspeed_config": None
}


# ========================================================================
# Step 7: Main Training Loop
# ========================================================================

def main():
    """
    Main function that orchestrates the entire training process.
    
    TRAINING WORKFLOW:
    1. Setup: Initialize distributed training, load model
    2. Prepare data: Load and tokenize dataset
    3. Choose strategy: Data Parallelism OR DeepSpeed/ZeRO
    4. Train: Run training epochs
    5. Cleanup: Free resources
    """
    
    # ----------------------------------------------------------------
    # Step 7.1: Load Configuration
    # ----------------------------------------------------------------
    class Args:
        """Simple class to hold configuration values."""
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

    # ----------------------------------------------------------------
    # Step 7.2: Initialize Distributed Training
    # ----------------------------------------------------------------
    setup_seed(args.seed)  # For reproducibility
    rank, local_rank, world_size, device = setup_distributed()

    model_info = MODEL_CONFIGS[args.model]

    # Print training configuration (only from rank 0 to avoid spam)
    if rank == 0:
        print("=" * 80)
        print(f"🚀 Fine-tuning GPT-2: {args.model}")
        print(f"   Model size: {model_info['params']} parameters")
        print(f"   Strategy: {args.strategy.upper()}")
        print("=" * 80)
        print(f"📊 Training Setup:")
        print(f"   GPUs: {world_size}")
        print(f"   Batch size per GPU: {args.batch_size}")
        print(f"   Total batch size: {args.batch_size * world_size}")
        print(f"   Epochs: {args.epochs}")
        print(f"   Training samples: {args.num_samples}")
        print("=" * 80)

    # ----------------------------------------------------------------
    # Step 7.3: Load Model and Tokenizer
    # ----------------------------------------------------------------
    if rank == 0:
        print("\n📥 Step 1: Loading pre-trained model from HuggingFace...")
        print(f"   Downloading {model_info['name']} (this may take a minute)...")

    # Load tokenizer (converts text ↔ numbers)
    tokenizer = GPT2Tokenizer.from_pretrained(model_info["name"])
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token by default

    # Load pre-trained model
    model = GPT2LMHeadModel.from_pretrained(model_info["name"])

    if rank == 0:
        # Count and display model size
        num_params = count_parameters(model)
        print(f"✅ Model loaded successfully!")
        print(f"   Total parameters: {num_params:,} ({num_params/1e6:.1f} million)")
        
        # Estimate memory requirements (rough calculation)
        # Each parameter in FP16 = 2 bytes
        param_memory_gb = num_params * 2 / 1e9
        print(f"\n💾 Memory Estimate (FP16 precision):")
        print(f"   Model parameters: {param_memory_gb:.2f} GB")
        print(f"   Gradients (same size as params): {param_memory_gb:.2f} GB")
        print(f"   Optimizer states (Adam stores 2 copies): {param_memory_gb * 6:.2f} GB")
        print(f"   ─────────────────────────────────")
        print(f"   Total per GPU (without ZeRO): ~{param_memory_gb * 9:.2f} GB")
        print(f"   (This is why we need ZeRO for large models!)")

    # ----------------------------------------------------------------
    # Step 7.4: Prepare Dataset
    # ----------------------------------------------------------------
    if rank == 0:
        print(f"\n� Step 2: Preparing WikiText-2 dataset...")

    dataset = prepare_dataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_samples=args.num_samples
    )
    
    if rank == 0:
        print(f"✅ Dataset ready!")
        print(f"   Total samples: {len(dataset)}")
        print(f"   Samples per GPU: {len(dataset) // world_size}")

    # ----------------------------------------------------------------
    # Step 7.5: Setup Training Strategy
    # ----------------------------------------------------------------
    
    if args.strategy == "dp":
        # ============================================================
        # DATA PARALLELISM (Baseline Strategy)
        # ============================================================
        if rank == 0:
            print(f"\n⚙️  Step 3: Setting up Data Parallelism (DDP)...")
            print(f"   Each GPU will hold a FULL copy of the model")
            print(f"   Memory usage: HIGH (not recommended for large models)")
        
        # Move model to GPU
        model = model.to(device)

        # Wrap model in DistributedDataParallel if using multiple GPUs
        if world_size > 1:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank
            )
            if rank == 0:
                print(f"   ✅ Model wrapped in DDP for {world_size} GPUs")

        # Create data sampler (splits data across GPUs)
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank
        ) if world_size > 1 else None
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler
        )

        # Create optimizer (AdamW is Adam with weight decay)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-5,           # Learning rate
            weight_decay=0.01  # L2 regularization
        )
        
        # Learning rate scheduler (warmup + linear decay)
        total_steps = len(dataloader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,      # Gradual warmup for stability
            num_training_steps=total_steps
        )

        if rank == 0:
            print(f"\n✅ Data Parallelism setup complete!")
            print(f"   Optimizer: AdamW (lr=5e-5)")
            print(f"   Scheduler: Linear warmup + decay")
            print(f"   Total training steps: {total_steps}")
            print_memory_stats(device, rank, prefix="\n   ")

        # ============================================================
        # START TRAINING (Data Parallelism)
        # ============================================================
        if rank == 0:
            print("\n" + "=" * 80)
            print("🚂 Starting Training Loop (Data Parallelism)")
            print("=" * 80)

        # Train for multiple epochs
        for epoch in range(1, args.epochs + 1):
            # Set epoch for sampler (ensures different data order each epoch)
            if sampler:
                sampler.set_epoch(epoch)

            # Reset peak memory stats to monitor this epoch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            
            # Train for one epoch
            train_epoch_ddp(model, dataloader, optimizer, scheduler, device, rank, epoch)

            # Print memory usage after epoch
            if rank == 0:
                print_memory_stats(device, rank, prefix="   ")
                print()

    else:
        # ============================================================
        # DEEPSPEED / ZeRO STRATEGIES
        # ============================================================
        if rank == 0:
            print(f"\n⚙️  Step 3: Setting up DeepSpeed ({args.strategy.upper()})...")
            
            # Explain which strategy is being used
            if args.strategy == "zero1":
                print(f"   ZeRO Stage 1: Shard optimizer states only")
                print(f"   Memory savings: ~4× reduction")
            elif args.strategy == "zero2":
                print(f"   ZeRO Stage 2: Shard optimizer + gradients")
                print(f"   Memory savings: ~8× reduction")
            elif args.strategy == "zero3":
                print(f"   ZeRO Stage 3: Shard EVERYTHING (params + grads + optimizer)")
                print(f"   Memory savings: ~{world_size}× reduction")
            elif args.strategy == "offload":
                print(f"   ZeRO-Offload: Use CPU RAM for optimizer states")
                print(f"   Allows training larger models with slower speed")
            elif args.strategy == "infinity":
                print(f"   ZeRO-Infinity: Use NVMe storage + CPU + GPU")
                print(f"   Can train MASSIVE models (100B+ parameters)")
        
        # Auto-select DeepSpeed config file if not specified
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
                print(f"   Using config: {args.deepspeed}")

        if rank == 0:
            print(f"\n   Initializing DeepSpeed...")

        # Initialize DeepSpeed
        # DeepSpeed will automatically:
        # - Shard the model across GPUs (based on ZeRO stage)
        # - Setup optimizers with ZeRO
        # - Create dataloaders
        # - Handle all communication between GPUs
        model_engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=dataset,
            config=args.deepspeed,
        )

        if rank == 0:
            print(f"\n✅ DeepSpeed initialization complete!")
            print(f"   Model sharded across {world_size} GPUs")
            print(f"   Strategy: {args.strategy.upper()}")
            print_memory_stats(model_engine.device, rank, prefix="\n   ")

        # ============================================================
        # START TRAINING (DeepSpeed/ZeRO)
        # ============================================================
        if rank == 0:
            print("\n" + "=" * 80)
            print(f"🚂 Starting Training Loop ({args.strategy.upper()})")
            print("=" * 80)

        # Train for multiple epochs
        for epoch in range(1, args.epochs + 1):
            # Reset peak memory stats
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(model_engine.device)
            
            # Train for one epoch
            train_epoch_deepspeed(model_engine, training_dataloader, rank, epoch)

            # Print memory usage after epoch
            if rank == 0:
                print_memory_stats(model_engine.device, rank, prefix="   ")
                print()

    # ----------------------------------------------------------------
    # Step 7.6: Training Complete!
    # ----------------------------------------------------------------
    if rank == 0:
        print("\n" + "=" * 80)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 80)
        print(f"\n� Summary:")
        print(f"   Model: {args.model} ({model_info['params']} parameters)")
        print(f"   Strategy: {args.strategy.upper()}")
        print(f"   Epochs trained: {args.epochs}")
        print(f"   Samples used: {args.num_samples}")
        print(f"   GPUs used: {world_size}")
        print("\n💡 What you learned:")
        print(f"   - How {args.strategy.upper()} reduces memory usage")
        print("   - How to fine-tune a real language model")
        print("   - How distributed training works across multiple GPUs")
        print("\n🚀 Next steps:")
        print("   - Try different strategies in CONFIG to compare")
        print("   - Increase epochs for better model performance")
        print("   - Try larger models (gpt2-large, gpt2-xl)")
        print("   - Use your own dataset instead of WikiText-2")
        print("=" * 80)

    # Cleanup distributed training
    cleanup_distributed()


# ========================================================================
# Entry Point
# ========================================================================

if __name__ == "__main__":
    """
    This is where the program starts when you run:
        python real_model_example.py
    
    For multi-GPU training, use one of these launchers:
        torchrun --nproc_per_node=4 real_model_example.py  (for Data Parallelism)
        deepspeed --num_gpus=4 real_model_example.py       (for ZeRO strategies)
    
    LEARNING TIP:
    Read through this code from top to bottom. Each section is clearly
    marked with comments explaining what it does and why. Start with the
    CONFIG dictionary at the top to change settings!
    """
    main()
