# Distributed Training with Real Models: GPT-2 Fine-tuning

> **⚡ Hands-On Tutorial**: Fine-tune GPT-2 models using Data Parallelism and ZeRO
> 
> **Time**: 30-60 minutes
> 
> **Prerequisites**: Multi-GPU cluster, PyTorch, DeepSpeed, HuggingFace Transformers

---

## 🎯 What You'll Learn

Fine-tune **real open-source models** (GPT-2 from HuggingFace) using different distributed training strategies:

1. **Data Parallelism (DP)** - Simple baseline, replicate model across GPUs
2. **ZeRO Stage 1** - Shard optimizer states (4× memory reduction)
3. **ZeRO Stage 2** - Shard optimizer + gradients (8× reduction)
4. **ZeRO Stage 3** - Shard everything (N× reduction, N = GPU count)
5. **ZeRO-Offload** - Use CPU memory for optimizer
6. **ZeRO-Infinity** - Use NVMe for massive models

Each example shows:
- ✅ **Real models** from HuggingFace (GPT-2 small/medium/large/xl)
- ✅ **Real datasets** (WikiText-2)
- ✅ **Actual memory usage** on your hardware
- ✅ **Training speed** comparisons

---

## � Available Models

| Model | Parameters | Memory (FP16) | Best For |
|-------|-----------|---------------|----------|
| `gpt2` | 124M | ~1.1 GB | Testing, single GPU |
| `gpt2-medium` | 355M | ~3.2 GB | Multi-GPU baseline |
| `gpt2-large` | 774M | ~7.0 GB | ZeRO-2/3 required |
| `gpt2-xl` | 1.5B | ~13.5 GB | ZeRO-3 required |

*Memory = parameters only. Total training memory is ~9× higher (gradients + optimizer).*

---

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install deepspeed transformers datasets accelerate psutil

# Or use requirements.txt
pip install -r requirements.txt

# Verify installation
ds_report
nvidia-smi
```

### First Run: Test Installation (1 GPU)
```bash
# Quick test with smallest model
python real_model_example.py \
    --model gpt2 \
    --strategy dp \
    --batch_size 4 \
    --epochs 1 \
    --num_samples 100
```

**Expected:** Downloads GPT-2 (124M), trains on 100 samples, shows memory usage.

---

### Multi-GPU: Data Parallelism (4 GPUs)
```bash
# Fine-tune GPT-2 Medium with standard data parallelism
torchrun --nproc_per_node=4 real_model_example.py \
    --model gpt2-medium \
    --strategy dp \
    --batch_size 4 \
    --epochs 3
```

**What it does:** 
- Each GPU holds full model (355M params)
- Each GPU processes different data
- Gradients synchronized after each batch

**Memory:** ~28 GB per GPU  
**Speed:** ~90 samples/sec (baseline)

---

### ZeRO Stage 2: Memory Savings (4 GPUs)
```bash
# Shard optimizer + gradients across GPUs
deepspeed --num_gpus=4 real_model_example.py \
    --model gpt2-medium \
    --strategy zero2 \
    --batch_size 4 \
    --epochs 3
```

**What it does:**
- Model parameters replicated on each GPU
- Optimizer states sharded (1/4 per GPU)
- Gradients sharded (1/4 per GPU)

**Memory:** ~18 GB per GPU (36% reduction!)  
**Speed:** ~90 samples/sec (same as DP!)

---

### ZeRO Stage 3: Maximum Savings (4 GPUs)
```bash
# Shard everything: params + grads + optimizer
deepspeed --num_gpus=4 real_model_example.py \
    --model gpt2-large \
    --strategy zero3 \
    --batch_size 2 \
    --epochs 3
```

**What it does:**
- Everything sharded across GPUs (1/4 each)
- All-gather parameters for forward/backward
- Can train 4× larger models!

**Memory:** ~10 GB per GPU (64% reduction!)  
**Speed:** ~80 samples/sec (10% slower)

---

### Compare All Strategies
```bash
# Automatically run DP, ZeRO-2, ZeRO-3 and compare
bash compare_strategies.sh gpt2-medium

# Results saved to comparison_results_gpt2-medium.txt
cat comparison_results_gpt2-medium.txt
```

---

## 📊 Example Results

### GPT-2 Medium (355M params) on 4× A100 GPUs:

| Strategy | Memory/GPU | Speed | Notes |
|----------|------------|-------|-------|
| **Data Parallel** | 28 GB | 90 samples/sec | Baseline |
| **ZeRO Stage 2** | 18 GB | 90 samples/sec | 36% memory saved |
| **ZeRO Stage 3** | 10 GB | 80 samples/sec | 64% memory saved |

**Key insight**: ZeRO-3 uses **64% less memory** with only **10% slower** training!

---

## 🎯 How to Choose a Strategy

| **Situation** | **Use This** | **Command Flag** |
|---------------|--------------|------------------|
| Model fits on single GPU | Data Parallel | `--strategy dp` |
| Need to fit larger model | ZeRO Stage 2 | `--strategy zero2` |
| Need maximum memory savings | ZeRO Stage 3 | `--strategy zero3` |
| Model doesn't fit in GPU memory | ZeRO-Offload | `--strategy offload` |
| Extremely large models (>10B) | ZeRO-Infinity | `--strategy infinity` |

---

## � Understanding the Code

### Key Functions in `real_model_example.py`

**1. setup_distributed()**: Initialize multi-GPU environment
```python
torch.distributed.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
```

**2. prepare_dataset()**: Load WikiText-2 from HuggingFace
```python
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
tokenized = dataset.map(tokenize_function, batched=True)
```

**3. train_epoch_ddp()**: Data Parallel training loop
- Model wrapped in `DistributedDataParallel`
- Gradients synchronized automatically after backward()

**4. train_epoch_deepspeed()**: ZeRO training loop  
- Model wrapped by DeepSpeed engine
- Memory sharding handled automatically
- Use `model_engine.backward(loss)` instead of `loss.backward()`

---

## 🚀 Advanced Usage

### Multi-Node Training (SLURM)

Create a SLURM job script (`train_gpt2.slurm`):
```bash
#!/bin/bash
#SBATCH --job-name=gpt2-zero3
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00

# Load modules
module load cuda/11.8
module load python/3.10

# Activate environment
source venv/bin/activate

# Run with DeepSpeed multi-node
deepspeed --num_nodes=2 \
          --num_gpus=4 \
          --master_addr=$SLURM_NODELIST \
          real_model_example.py \
          --model gpt2-large \
          --strategy zero3 \
          --batch_size 2 \
          --epochs 10
```

Submit: `sbatch train_gpt2.slurm`

---

### Custom DeepSpeed Configuration

Modify `ds_config_stage3.json` for your needs:
```json
{
  "zero_optimization": {
    "stage": 3,
    "stage3_prefetch_bucket_size": 5e8,  // Increase for faster training
    "stage3_param_persistence_threshold": 1e6,  // Lower = more memory savings
    "overlap_comm": true  // Enable communication overlap
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16
  }
}
```

---

### Memory Profiling

Add `--profile_memory` to see detailed breakdown:
```bash
python real_model_example.py \
    --model gpt2-large \
    --strategy zero3 \
    --batch_size 2 \
    --profile_memory
```

**Output**:
```
GPU 0 Memory:
  Allocated: 8.2 GB
  Reserved: 9.1 GB
  Peak: 10.5 GB
```

---

## 📚 Further Reading

- **REAL_MODELS_GUIDE.md**: Comprehensive guide with more GPT-2 examples and tips
- **CLUSTER_QUICKSTART.md**: Quick setup guide for your cluster
- **DeepSpeed Documentation**: [deepspeed.ai](https://www.deepspeed.ai/)
- **ZeRO Paper**: [arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054)
- **HuggingFace Transformers**: [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: 
- Reduce `--batch_size`
- Switch to ZeRO-3: `--strategy zero3`
- Use gradient accumulation steps in DeepSpeed config

### Issue: Slow Training
**Solution**:
- Check `overlap_comm: true` in DeepSpeed config
- Increase `stage3_prefetch_bucket_size` 
- Verify fast interconnect (InfiniBand/NVLink)

### Issue: NCCL Timeout
**Solution**:
```bash
export NCCL_TIMEOUT=3600  # Increase timeout
export NCCL_DEBUG=INFO    # Enable debug logging
```

### Issue: Model Download Fails
**Solution**:
```bash
# Pre-download models
python -c "from transformers import GPT2LMHeadModel; GPT2LMHeadModel.from_pretrained('gpt2')"

# Or set cache directory
export HF_HOME=/path/to/cache
```

---

## 🎓 What You Learned

✅ **Data Parallelism**: Simple multi-GPU training (baseline)  
✅ **ZeRO Stage 2**: Shard optimizer + gradients (36% memory savings)  
✅ **ZeRO Stage 3**: Shard everything (64% memory savings)  
✅ **ZeRO-Offload**: Use CPU memory when GPU is full  
✅ **ZeRO-Infinity**: Use NVMe for massive models  
✅ **Real Models**: Fine-tuned GPT-2 on WikiText-2  
✅ **Benchmarking**: Compared strategies on same hardware  

---

## 🚀 Next Steps

1. **Try different models**: `--model gpt2-xl` (1.5B params)
2. **Increase scale**: Run on 8 GPUs or multiple nodes
3. **Add monitoring**: Integrate with Weights & Biases (`wandb`)
4. **Custom datasets**: Replace WikiText-2 with your own data
5. **Production training**: Add checkpointing, evaluation, early stopping

See **REAL_MODELS_GUIDE.md** for more experiments!

---

## 📖 Additional Resources

- **Files in this tutorial**:
  - `real_model_example.py` - Main training script
  - `ds_config_stage*.json` - DeepSpeed configurations
  - `REAL_MODELS_GUIDE.md` - Detailed GPT-2 guide
  - `CLUSTER_QUICKSTART.md` - Quick cluster setup
  - `compare_strategies.sh` - Automated benchmarking

- **External Resources**:
  - [DeepSpeed Official Docs](https://www.deepspeed.ai/)
  - [ZeRO Paper (arXiv)](https://arxiv.org/abs/1910.02054)
  - [HuggingFace Transformers](https://huggingface.co/docs/transformers)
  - [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

---

**Happy Training! 🚀**

---

*This tutorial was designed for hands-on learning with real models. For questions or issues, see REAL_MODELS_GUIDE.md for troubleshooting tips.*
