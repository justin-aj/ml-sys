# Real Model Examples - Quick Reference 🚀

This guide shows you how to fine-tune **real open-source models** (GPT-2) using different distributed training strategies.

---

## 🎯 Why Use Real Models?

✅ **Realistic performance** - See actual memory usage  
✅ **Meaningful results** - Train on real datasets (WikiText)  
✅ **Easy to run** - Models download automatically from HuggingFace  
✅ **Multiple sizes** - Test from 124M to 1.5B parameters  

---

## 📊 Available Models

| Model | Parameters | Hidden Size | Layers | Memory (FP16) | Best Strategy |
|-------|-----------|-------------|--------|---------------|---------------|
| `gpt2` | 124M | 768 | 12 | ~1.1 GB | Data Parallel |
| `gpt2-medium` | 355M | 1024 | 24 | ~3.2 GB | Data Parallel or ZeRO-1 |
| `gpt2-large` | 774M | 1280 | 36 | ~7.0 GB | ZeRO-2 or ZeRO-3 |
| `gpt2-xl` | 1.5B | 1600 | 48 | ~13.5 GB | ZeRO-3 |

*Memory = Parameters only (FP16). Total training memory is ~9× higher (includes gradients + optimizer).*

---

## 🚀 Quick Start

### Example 1: GPT-2 Medium with Data Parallelism
```bash
# 4 GPUs, batch size 4 per GPU
torchrun --nproc_per_node=4 real_model_example.py \
    --model gpt2-medium \
    --strategy dp \
    --batch_size 4 \
    --epochs 3
```

**What happens:**
- Downloads GPT-2 Medium (355M params) from HuggingFace
- Loads WikiText-2 dataset
- Fine-tunes on 1000 samples
- Shows memory usage per GPU
- Prints training throughput

**Expected memory:** ~30 GB per GPU (355M params × 9 with optimizer)

---

### Example 2: GPT-2 Large with ZeRO Stage 3
```bash
# 4 GPUs, ZeRO-3 for memory savings
deepspeed --num_gpus=4 real_model_example.py \
    --model gpt2-large \
    --strategy zero3 \
    --batch_size 2
```

**What happens:**
- Shards 774M parameter model across 4 GPUs
- Each GPU only holds 1/4 of the model
- Memory per GPU: ~15 GB (vs ~63 GB without ZeRO!)
- Slightly slower due to all-gather communication

**Expected memory:** ~15 GB per GPU (4× reduction from ZeRO-3)

---

### Example 3: Compare All Strategies
```bash
# Automatically run Data Parallel, ZeRO-2, and ZeRO-3
bash compare_strategies.sh gpt2-medium
```

**Output:** Results saved to `comparison_results_gpt2-medium.txt`

Shows:
- Memory usage for each strategy
- Training speed (samples/sec)
- Loss curves
- Peak memory allocation

---

## 📋 All Command-Line Options

```bash
python real_model_example.py \
    --model <MODEL_NAME> \          # gpt2, gpt2-medium, gpt2-large, gpt2-xl
    --strategy <STRATEGY> \         # dp, zero1, zero2, zero3, offload, infinity
    --batch_size <SIZE> \           # Batch size per GPU (default: 4)
    --epochs <NUM> \                # Number of epochs (default: 3)
    --max_length <LEN> \            # Max sequence length (default: 512)
    --num_samples <NUM> \           # Training samples (default: 1000)
    --deepspeed <CONFIG>            # DeepSpeed config file (auto-selected if not provided)
```

---

## 🎓 Learning Path

### Step 1: Start Small (GPT-2 Small)
```bash
# 1 GPU, simple data parallel
python real_model_example.py --model gpt2 --strategy dp --batch_size 8

# What to observe:
# - Training speed baseline
# - Memory usage: ~10 GB
# - Loss should decrease steadily
```

### Step 2: Scale Up (GPT-2 Medium)
```bash
# 4 GPUs, data parallel
torchrun --nproc_per_node=4 real_model_example.py \
    --model gpt2-medium --strategy dp --batch_size 4

# What to observe:
# - 4× throughput (parallelism working!)
# - Memory usage: ~30 GB per GPU
# - Each GPU sees different batches
```

### Step 3: Add ZeRO (GPT-2 Medium)
```bash
# Compare Stage 2 vs Stage 3
deepspeed --num_gpus=4 real_model_example.py \
    --model gpt2-medium --strategy zero2 --batch_size 4

deepspeed --num_gpus=4 real_model_example.py \
    --model gpt2-medium --strategy zero3 --batch_size 4

# What to observe:
# - Stage 2: ~20 GB per GPU (vs 30 GB baseline)
# - Stage 3: ~12 GB per GPU (2.5× reduction!)
# - Speed: Stage 3 slightly slower (all-gather overhead)
```

### Step 4: Push Limits (GPT-2 Large/XL)
```bash
# GPT-2 Large requires ZeRO-3
deepspeed --num_gpus=4 real_model_example.py \
    --model gpt2-large --strategy zero3 --batch_size 2

# GPT-2 XL requires ZeRO-3 + smaller batch
deepspeed --num_gpus=4 real_model_example.py \
    --model gpt2-xl --strategy zero3 --batch_size 1

# What to observe:
# - Without ZeRO: Would OOM (Out of Memory) ❌
# - With ZeRO-3: Fits comfortably ✅
# - This is how you train billion-param models!
```

---

## 💡 Tips & Tricks

### Reduce Memory Usage
```bash
# 1. Smaller batch size
--batch_size 1

# 2. Shorter sequences
--max_length 256

# 3. Higher ZeRO stage
--strategy zero3  # instead of zero2

# 4. Enable activation checkpointing (edit DeepSpeed config)
"activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true
}
```

### Increase Training Speed
```bash
# 1. Larger batch size (if memory allows)
--batch_size 8

# 2. Lower ZeRO stage (less communication)
--strategy zero2  # instead of zero3

# 3. More GPUs (linear scaling!)
deepspeed --num_gpus=8 ...

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
