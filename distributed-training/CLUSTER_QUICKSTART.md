# 🎯 CLUSTER QUICK START - Run This First!

This is your **go-to guide** for running the distributed training tutorial on your college cluster.

---

## ✅ Pre-Flight Checklist

Before running anything, make sure you have:
- [ ] Access to multi-GPU cluster (at least 2 GPUs)
- [ ] Python 3.8+
- [ ] CUDA 11.0+ (check with `nvidia-smi`)
- [ ] Internet access (to download models/datasets)

---

## 🚀 Step-by-Step Setup

### Step 1: Navigate to Tutorial Directory
```bash
cd distributed-training
```

### Step 2: Install Dependencies
```bash
# Option 1: Using pip (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install deepspeed transformers datasets accelerate psutil

# Option 2: Using requirements.txt
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
ds_report  # Check DeepSpeed environment
```

### Step 3: Check GPU Availability
```bash
# How many GPUs do you have?
nvidia-smi --query-gpu=count,name,memory.total --format=csv

# Set this variable for later use
export NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
echo "Found $NUM_GPUS GPUs"
```

---

## 🎯 Recommended First Run: Real Model Example

**Best starting point!** Uses actual GPT-2 from HuggingFace.

### Quick Test (1 GPU)
```bash
# Test installation with smallest model (124M params)
python real_model_example.py \
    --model gpt2 \
    --strategy dp \
    --batch_size 4 \
    --epochs 1 \
    --num_samples 100
```

**Expected output:**
```
🚀 Fine-tuning gpt2 (124M parameters)
📋 Strategy: DP
================================================================================
📥 Loading model and tokenizer...
✅ Model loaded: 124,439,808 parameters (124.4M)
📦 Loading dataset (WikiText-2)...
✅ Dataset ready: 100 samples
🚂 Starting Training (Data Parallelism)
Epoch 1 | Batch 0/25 | Loss: 3.4521 | LR: 5.00e-05
...
✅ Training Complete!
GPU Memory: 1.23 GB allocated | 1.50 GB reserved | 1.45 GB peak
```

### Multi-GPU Test (4 GPUs)
```bash
# Test distributed training
torchrun --nproc_per_node=4 real_model_example.py \
    --model gpt2-medium \
    --strategy dp \
    --batch_size 4 \
    --epochs 1 \
    --num_samples 200
```

**What to observe:**
- All 4 GPUs should be utilized (~25% each in `nvidia-smi`)
- Training should be ~4× faster than single GPU
- Each GPU uses ~7 GB memory

### ZeRO Test (Memory Savings)
```bash
# Compare memory usage with ZeRO-3
deepspeed --num_gpus=4 real_model_example.py \
    --model gpt2-medium \
    --strategy zero3 \
    --batch_size 4 \
    --epochs 1 \
    --num_samples 200
```

**What to observe:**
- GPU memory drops from ~7 GB to ~3 GB per GPU (2× reduction!)
- Training slightly slower (~10%) but much more memory efficient
- Can now train 2× larger models with same hardware

---

## 📊 Full Comparison (Recommended)

Run all strategies on same model and compare:

```bash
# Automatic comparison
bash compare_strategies.sh gpt2-medium

# Check results
cat comparison_results_gpt2-medium.txt
```

This will:
1. Run Data Parallelism
2. Run ZeRO Stage 2
3. Run ZeRO Stage 3
4. Save memory & speed comparison

---

## 🎓 Understanding Different Strategies

You can experiment with different strategies using the same script:

### Data Parallelism (Baseline)
```bash
torchrun --nproc_per_node=$NUM_GPUS real_model_example.py \
    --model gpt2-medium \
    --strategy dp \
    --batch_size 4 \
    --epochs 3
```

### ZeRO Stage 2 (Memory Optimized)
```bash
deepspeed --num_gpus=$NUM_GPUS real_model_example.py \
    --model gpt2-medium \
    --strategy zero2 \
    --batch_size 4 \
    --epochs 3
```

### ZeRO Stage 3 (Maximum Savings)
```bash
deepspeed --num_gpus=$NUM_GPUS real_model_example.py \
    --model gpt2-large \
    --strategy zero3 \
    --batch_size 2 \
    --epochs 3
```

### ZeRO-Offload (CPU Memory)
```bash
deepspeed --num_gpus=$NUM_GPUS real_model_example.py \
    --model gpt2-large \
    --strategy offload \
    --batch_size 2 \
    --epochs 3
```

### ZeRO-Infinity (NVMe Storage)
```bash
# Create offload directory first
mkdir -p /tmp/nvme_offload

deepspeed --num_gpus=$NUM_GPUS real_model_example.py \
    --model gpt2-xl \
    --strategy infinity \
    --batch_size 1 \
    --epochs 3
```

---

## 🐛 Troubleshooting

### Issue 1: "ModuleNotFoundError: No module named 'deepspeed'"
```bash
# Solution: Install DeepSpeed
pip install deepspeed

# If that fails, try building from source:
pip install deepspeed --global-option="build_ext" --global-option="-j8"
```

### Issue 2: "CUDA out of memory"
```bash
# Solution: Reduce batch size
--batch_size 2  # or even 1

# Or use higher ZeRO stage
--strategy zero3  # instead of dp or zero2
```

### Issue 3: "NCCL error" in multi-GPU setup
```bash
# Solution: Set environment variables
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # Replace with your network interface

# Check with:
ifconfig  # or ip addr
```

### Issue 4: Can't download models (no internet on compute nodes)
```bash
# Solution: Download on head node first
python -c "from transformers import GPT2LMHeadModel; GPT2LMHeadModel.from_pretrained('gpt2-medium')"
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-2-raw-v1')"

# Models cached to ~/.cache/huggingface/
# Copy cache to compute nodes if needed
```

### Issue 5: "Too many open files" error
```bash
# Solution: Increase file limit
ulimit -n 65536
```

---

## 📈 What to Expect

### GPT-2 Medium (355M params) on 4× V100 GPUs:

| Strategy | Memory/GPU | Speed | Time (1 epoch, 1000 samples) |
|----------|------------|-------|------------------------------|
| Data Parallel | ~28 GB | ~90 samples/sec | ~11 sec |
| ZeRO Stage 2 | ~18 GB | ~90 samples/sec | ~11 sec |
| ZeRO Stage 3 | ~10 GB | ~80 samples/sec | ~12.5 sec |

**Key insight:** ZeRO-3 uses **64% less memory** with only **10% speed loss**!

---

## 🎯 Recommended Learning Path

### Day 1: Understand Basics
1. Run `real_model_example.py` with Data Parallelism
2. Monitor with `watch -n 1 nvidia-smi` in another terminal
3. Understand the output (memory, throughput, loss)

### Day 2: Explore ZeRO
1. Run same model with `--strategy zero3`
2. Compare memory usage (should be ~3× less!)
3. Run `compare_strategies.sh` for side-by-side comparison

### Day 3: Scale Up
1. Try larger model: `--model gpt2-large` with ZeRO-3
2. Try even larger: `--model gpt2-xl` (1.5B params)
3. Experiment with ZeRO-Offload and ZeRO-Infinity

### Day 4: Production Ready
1. Add checkpointing and resuming
2. Integrate monitoring (wandb, tensorboard)
3. Try your own dataset instead of WikiText-2

---

## 📚 Additional Resources

- **README.md**: Main tutorial documentation
- **REAL_MODELS_GUIDE.md**: Comprehensive GPT-2 examples and tips
- **compare_strategies.sh**: Automated benchmarking script
- DeepSpeed Docs: https://www.deepspeed.ai/
- ZeRO Paper: https://arxiv.org/abs/1910.02054

---

## 📝 Sample Batch Script (SLURM)

If your cluster uses SLURM:

```bash
#!/bin/bash
#SBATCH --job-name=distributed-training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=training_%j.log

# Load modules (adjust for your cluster)
module load python/3.9
module load cuda/11.8

# Activate environment
source ~/venv/bin/activate

# Run training
deepspeed --num_gpus=4 real_model_example.py \
    --model gpt2-medium \
    --strategy zero3 \
    --batch_size 4 \
    --epochs 3
```

Submit with: `sbatch train.slurm`

---

## 🏆 Quick Wins

**Win 1: Train Your First Model (5 min)**
```bash
python real_model_example.py --model gpt2 --strategy dp --epochs 1 --num_samples 100
```

**Win 2: See ZeRO Memory Savings (10 min)**
```bash
bash compare_strategies.sh gpt2-medium
```

**Win 3: Train Largest Possible Model (15 min)**
```bash
deepspeed --num_gpus=4 real_model_example.py --model gpt2-xl --strategy zero3 --batch_size 1
```

---

## 🎓 Next Steps

After completing this tutorial:

1. **Read the docs:** 
   - `README.md` - Full tutorial
   - `REAL_MODELS_GUIDE.md` - Model-specific guide
   - `MEMORY_COMPARISON.md` - Visual memory breakdowns

2. **Experiment:**
   - Different model sizes
   - Different batch sizes
   - Multi-node training

3. **Apply to your research:**
   - Use your own dataset
   - Fine-tune for your task
   - Scale to larger models

---

## ✅ Success Criteria

You'll know you succeeded when:
- ✅ Can run GPT-2 training on multiple GPUs
- ✅ Understand memory usage differences between strategies
- ✅ Can choose the right strategy for your model size
- ✅ Can debug common issues (OOM, NCCL errors)

---

**Start here:**
```bash
python real_model_example.py --model gpt2 --strategy dp --epochs 1
```

Then watch your GPU memory in another terminal:
```bash
watch -n 1 nvidia-smi
```

Good luck! 🚀
