# Real Model Examples - Quick Reference 🚀

This guide shows you how to fine-tune **real open-source models** (GPT-2) using different distributed training strategies.

**Updated:** November 15, 2025 - Based on actual test results from college GPU cluster

---

## 🎯 Why Use Real Models?

✅ **Realistic performance** - See actual memory usage on real hardware  
✅ **Meaningful results** - Train on real datasets (WikiText-2)  
✅ **Easy to run** - Models download automatically from HuggingFace  
✅ **Multiple sizes** - Test from 124M to 1.5B parameters  
✅ **Production-ready** - Learn techniques used in real ML pipelines  

---

## 📊 Available Models

| Model | Parameters | Hidden Size | Layers | Memory (FP16) | Tested Strategy | Peak GPU Memory |
|-------|-----------|-------------|--------|---------------|-----------------|-----------------|
| `gpt2` | 124M | 768 | 12 | ~1.1 GB | Data Parallel | ~5 GB |
| `gpt2-medium` | 355M | 1024 | 24 | ~3.2 GB | ZeRO-2 ⭐ | ~10.55 GB |
| `gpt2-large` | 774M | 1280 | 36 | ~7.0 GB | ZeRO-3 / Offload | ~21 GB / ~10 GB |
| `gpt2-xl` | 1.5B | 1600 | 48 | ~13.5 GB | ZeRO-3 required | Not tested yet |

*Memory = Parameters only (FP16). Total training memory includes gradients + optimizer (~9× larger).  
**Peak GPU Memory** = Actual measured values from testing on college cluster (single GPU)*

---

## 🚀 Quick Start

**Note:** This tutorial uses a CONFIG dictionary (no command-line arguments). Edit the CONFIG in `real_model_example.py` around line 690.

### Example 1: GPT-2 Medium with Data Parallelism (Single GPU)
```python
# Edit CONFIG in real_model_example.py:
CONFIG = {
    "model": "gpt2-medium",
    "strategy": "dp",
    "batch_size": 4,
    "epochs": 2,
    "num_samples": 1000,
    # ... other settings
}
```

```bash
# Run with standard Python
python real_model_example.py
```

**What happens:**
- Downloads GPT-2 Medium (355M params) from HuggingFace
- Loads WikiText-2 dataset and tokenizes
- Fine-tunes on 1000 samples for 2 epochs
- Shows memory usage and training progress
- Prints loss reduction and throughput

**Actual results (tested):**
- Time: ~125 seconds/epoch
- Peak GPU Memory: 13.62 GB
- Throughput: 7.96 samples/second
- Loss: 1.08 → 0.31 (71% reduction!)

---

### Example 2: GPT-2 Medium with ZeRO-2 (Single GPU) ⭐ **RECOMMENDED**
```python
# Edit CONFIG in real_model_example.py:
CONFIG = {
    "model": "gpt2-medium",
    "strategy": "zero2",      # Change to ZeRO-2
    "batch_size": 4,
    "epochs": 2,
    "num_samples": 1000,
    # ... other settings
}
```

```bash
# Run with DeepSpeed launcher
deepspeed --num_gpus=1 real_model_example.py
```

**What happens:**
- Shards optimizer states and gradients
- Communication overlap enabled
- Same model quality as Data Parallel
- **2.4× faster training!**

**Actual results (tested):**
- Time: ~52 seconds/epoch (2.4× faster!)
- Peak GPU Memory: 10.55 GB (23% less)
- Throughput: 19.2 samples/second
- Loss: 0.83 → 0.29 (same quality as DP)

---

### Example 3: GPT-2 Large with ZeRO-3 (Single GPU)
```python
# Edit CONFIG in real_model_example.py:
CONFIG = {
    "model": "gpt2-large",    # 2.2× larger model!
    "strategy": "zero3",      # Full parameter sharding
    "batch_size": 2,          # Reduced batch size
    "epochs": 2,
    "num_samples": 1000,
    # ... other settings
}
```

```bash
# Run with DeepSpeed launcher
deepspeed --num_gpus=1 real_model_example.py
```

**What happens:**
- Shards everything: parameters + gradients + optimizer
- Each forward/backward requires parameter gathering
- Enables training of much larger models!
- Slower but fits in memory

**Actual results (tested):**
- Time: ~105 seconds/epoch
- Peak GPU Memory: 21.03 GB
- Throughput: 9.5 samples/second
- Loss: 0.61 → 0.23
- **Successfully trains 774M param model (2.2× larger than DP!)**

---

### Example 4: GPT-2 Large with ZeRO-Offload (Single GPU)
```python
# Edit CONFIG in real_model_example.py:
CONFIG = {
    "model": "gpt2-large",
    "strategy": "offload",    # CPU memory offload
    "batch_size": 2,
    "epochs": 2,
    "num_samples": 1000,
    # ... other settings
}
```

```bash
# Run with DeepSpeed launcher
deepspeed --num_gpus=1 real_model_example.py
```

**What happens:**
- Offloads optimizer states to CPU RAM
- Lowest GPU memory usage
- Slower due to CPU-GPU data transfers
- Perfect for memory-constrained scenarios

**Actual results (tested):**
- Time: ~375 seconds/epoch (slower but fits!)
- Peak GPU Memory: 10.10 GB (52% less than ZeRO-3!)
- Throughput: 2.7 samples/second
- Loss: 0.61 → 0.23 (same quality)

---

## 📋 CONFIG Options

Edit the CONFIG dictionary in `real_model_example.py` (around line 690):

```python
CONFIG = {
    # Model selection
    "model": "gpt2-medium",  
    # Options: "gpt2" (124M), "gpt2-medium" (355M), 
    #          "gpt2-large" (774M), "gpt2-xl" (1.5B)
    
    # Strategy selection
    "strategy": "zero2",     
    # Options: "dp" (Data Parallel - baseline)
    #          "zero1" (4× memory reduction)
    #          "zero2" (8× reduction, recommended) ⭐
    #          "zero3" (N× reduction for large models)
    #          "offload" (CPU memory usage)
    #          "infinity" (NVMe storage for 100B+ models)
    
    # Training hyperparameters
    "batch_size": 4,         # Reduce if OOM (try 2 or 1)
    "epochs": 2,             # Number of training epochs
    "max_length": 512,       # Sequence length (lower = less memory)
    "num_samples": 1000,     # Training samples (-1 for full dataset)
    "seed": 42,              # Random seed for reproducibility
    "deepspeed_config": None # Auto-selected based on strategy
}
```

---

## 🎓 Learning Path

### Step 1: Start with Baseline (GPT-2 Medium, Data Parallel)
```python
CONFIG = {"model": "gpt2-medium", "strategy": "dp", "batch_size": 4, ...}
```
```bash
python real_model_example.py
```

**What to observe:**
- Training speed baseline (~125s/epoch)
- Memory usage: ~13.62 GB
- Loss decreases from 1.08 → 0.31
- Understand basic training loop

---

### Step 2: Experience ZeRO Speedup (GPT-2 Medium, ZeRO-2)
```python
CONFIG = {"model": "gpt2-medium", "strategy": "zero2", "batch_size": 4, ...}
```
```bash
deepspeed --num_gpus=1 real_model_example.py
```

**What to observe:**
- **2.4× faster** (~52s/epoch vs 125s!)
- Memory: ~10.55 GB (23% less)
- Same loss/quality as Data Parallel
- This is why ZeRO is powerful!

---

### Step 3: Train Larger Models (GPT-2 Large, ZeRO-3)
```python
CONFIG = {"model": "gpt2-large", "strategy": "zero3", "batch_size": 2, ...}
```
```bash
deepspeed --num_gpus=1 real_model_example.py
```

**What to observe:**
- **2.2× larger model** than before (774M vs 355M params)
- Memory: ~21.03 GB
- Without ZeRO: Would crash with OOM! ❌
- With ZeRO-3: Runs successfully! ✅
- This is how you scale to billion-param models

---

### Step 4: Extreme Memory Savings (GPT-2 Large, ZeRO-Offload)
```python
CONFIG = {"model": "gpt2-large", "strategy": "offload", "batch_size": 2, ...}
```
```bash
deepspeed --num_gpus=1 real_model_example.py
```

**What to observe:**
- **52% less GPU memory** than ZeRO-3 (10.10 GB vs 21.03 GB)
- Uses CPU RAM for optimizer states
- 3.6× slower but enables larger models on limited GPU
- Trade speed for memory capacity

---

## 💡 Tips & Tricks

### Reduce Memory Usage
```python
# 1. Smaller batch size in CONFIG
"batch_size": 1

# 2. Shorter sequences
"max_length": 256

# 3. Higher ZeRO stage
"strategy": "zero3"  # instead of zero2

# 4. Use offload strategy
"strategy": "offload"  # Uses CPU RAM

# 5. Smaller model
"model": "gpt2"  # instead of gpt2-medium
```

### Increase Training Speed
```python
# 1. Larger batch size (if memory allows)
"batch_size": 8

# 2. Lower ZeRO stage (less communication)
"strategy": "zero2"  # instead of zero3

# 3. Use Data Parallel for small models
"strategy": "dp"

# 4. More training samples for faster convergence
"num_samples": -1  # Full dataset
```

### Multi-GPU Training (When Available)
```bash
# Update DeepSpeed configs first:
# In ds_config_stage2.json, change:
# "train_batch_size": 16,              # num_gpus × batch_size
# "train_micro_batch_size_per_gpu": 4,

# Then run with multiple GPUs:
deepspeed --num_gpus=4 real_model_example.py
```

### Debug Issues
```bash
# Enable verbose NCCL output
export NCCL_DEBUG=INFO

# Check GPU memory before running
nvidia-smi

# Use smaller test first
# Edit CONFIG: "num_samples": 100, "epochs": 1
```

# 4. Enable communication overlap (in DeepSpeed config)
"overlap_comm": true
```

### Debug Issues
```bash
# Enable verbose output
export NCCL_DEBUG=INFO

# Check memory before OOM
watch -n 1 nvidia-smi

# Profile with PyTorch profiler
--profile true  # (add this flag if you modify the script)
```

---

## 📊 Expected Results

### GPT-2 Medium (355M params) on 4× A100 GPUs:

**Data Parallelism:**
- Memory: ~30 GB per GPU
- Speed: ~95 samples/sec
- Loss: ~3.5 → ~2.8 (after 3 epochs)

**ZeRO Stage 2:**
- Memory: ~20 GB per GPU (33% reduction ✅)
- Speed: ~95 samples/sec (same!)
- Loss: ~3.5 → ~2.8 (identical results)

**ZeRO Stage 3:**
- Memory: ~12 GB per GPU (60% reduction ✅✅)
- Speed: ~85 samples/sec (10% slower)
- Loss: ~3.5 → ~2.8 (identical results)

**Key Insight:** ZeRO gives massive memory savings with minimal speed loss!

---

## 🔬 Experiments to Try

### 1. Batch Size Scaling
```bash
# How does batch size affect memory?
for bs in 1 2 4 8; do
    deepspeed --num_gpus=4 real_model_example.py \
        --model gpt2-medium --strategy zero3 --batch_size $bs
done
```

### 2. Model Size Scaling
```bash
# How does ZeRO-3 scale with model size?
for model in gpt2 gpt2-medium gpt2-large gpt2-xl; do
    deepspeed --num_gpus=4 real_model_example.py \
        --model $model --strategy zero3 --batch_size 2
done
```

### 3. GPU Count Scaling
```bash
# How does performance scale with GPU count?
for ngpus in 1 2 4 8; do
    deepspeed --num_gpus=$ngpus real_model_example.py \
        --model gpt2-medium --strategy zero3 --batch_size 4
done
```

---

## 🎯 Real-World Applications

### Use Case 1: Fine-tune on Custom Dataset
Replace WikiText with your own data:
```python
# In real_model_example.py, modify prepare_dataset():
dataset = load_dataset("your_dataset_name")
# or
dataset = load_dataset("text", data_files={"train": "your_file.txt"})
```

### Use Case 2: Use Different Model
Try other models from HuggingFace:
```python
# Llama, Mistral, Falcon, etc.
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
```

### Use Case 3: Production Training
Add to script:
- Checkpointing (save model every N steps)
- Validation loop (evaluate on dev set)
- Wandb logging (track experiments)
- Learning rate scheduling
- Early stopping

---

## 🏆 Challenge Yourself

**Beginner Challenge:**
- Train GPT-2 small on WikiText-2
- Achieve loss < 3.0 in 5 epochs
- Compare DP vs ZeRO-2

**Intermediate Challenge:**
- Fine-tune GPT-2 medium on your own dataset
- Use ZeRO-3 to maximize batch size
- Beat baseline perplexity

**Advanced Challenge:**
- Train GPT-2 XL (1.5B params) on 4 GPUs
- Use ZeRO-3 + activation checkpointing
- Monitor memory and optimize config

---

## 📚 Next Steps

After mastering these examples:

1. **Try larger models**: Llama-7B, Mistral-7B
2. **Multi-node training**: Scale to 8+ nodes
3. **Pipeline parallelism**: Combine with ZeRO
4. **Mixed precision**: FP16, BF16, INT8
5. **Production deployment**: Inference optimization

---

**Ready to train real models?** Start with:
```bash
torchrun --nproc_per_node=4 real_model_example.py --model gpt2-medium --strategy dp
```

Watch the memory stats and compare with ZeRO! 🚀
