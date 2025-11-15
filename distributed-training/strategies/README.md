# Distributed Training Strategies

This directory contains implementations and configurations for 6 different distributed training strategies.

---

## 📁 Directory Structure

```
strategies/
├── 1_data_parallel/      - Baseline: Replicate model across GPUs
├── 2_zero_stage1/        - Shard optimizer states (4× memory reduction)
├── 3_zero_stage2/        - Shard optimizer + gradients (8× reduction) ⭐ RECOMMENDED
├── 4_zero_stage3/        - Shard everything (N× reduction)
├── 5_zero_offload/       - Use CPU memory for optimizer
└── 6_zero_infinity/      - Use NVMe storage for massive models
```

---

## 🎯 Quick Comparison

| Strategy | Memory Savings | Speed | Best For |
|----------|----------------|-------|----------|
| **1. Data Parallel** | None (baseline) | 1.0× | Small models, testing |
| **2. ZeRO Stage 1** | 29% less | **2.4× faster** | Production (optimizer only) |
| **3. ZeRO Stage 2** | 23% less | **2.4× faster** | **Production (recommended)** ⭐ |
| **4. ZeRO Stage 3** | Enables 2.2× larger | 1.0× | Large models that don't fit |
| **5. ZeRO-Offload** | 52% less GPU | 0.3× | Limited GPU memory |
| **6. ZeRO-Infinity** | Maximum savings | Slowest | Extremely large models |

---

## 🚀 How to Use

Each directory contains:
- **README.md** - Strategy explanation and usage
- **config.json** - DeepSpeed configuration (if applicable)
- **run.sh** - Quick run script

### Running a Strategy

```bash
# Example: Run ZeRO Stage 2 (recommended)
cd 3_zero_stage2
bash run.sh
```

Or use the main script from parent directory:

```bash
# From distributed-training/ directory
# Edit CONFIG in real_model_example.py:
CONFIG = {
    "strategy": "zero2",  # Choose: dp, zero1, zero2, zero3, offload, infinity
    # ...
}

# Run
deepspeed --num_gpus=1 real_model_example.py
```

---

## 📊 Tested Results (Single GPU - Nov 15, 2025)

### GPT-2 Medium (355M params):

| Strategy | Time/Epoch | GPU Memory | Speedup |
|----------|------------|------------|---------|
| Data Parallel | 125s | 13.62 GB | 1.0× |
| ZeRO-1 | 52s | 9.64 GB | **2.4×** |
| ZeRO-2 | 52s | 10.55 GB | **2.4×** ⭐ |

### GPT-2 Large (774M params):

| Strategy | Time/Epoch | GPU Memory | Notes |
|----------|------------|------------|-------|
| ZeRO-3 | 105s | 21.03 GB | 2.2× larger model |
| ZeRO-Offload | 375s | 10.10 GB | 52% less memory |

---

## 🎓 Learning Path

**Day 1:** Start with `1_data_parallel/` - understand baseline  
**Day 2:** Try `3_zero_stage2/` - see 2.4× speedup!  
**Day 3:** Experiment with `4_zero_stage3/` - train larger models  
**Day 4:** Test `5_zero_offload/` - extreme memory savings

---

## 📚 More Information

- **Main Tutorial**: See `../README.md`
- **Complete Results**: See `../TRAINING_RESULTS.md`
- **Advanced Guide**: See `../REAL_MODELS_GUIDE.md`
- **Cluster Setup**: See `../CLUSTER_QUICKSTART.md`

---

**Questions?** Check the README.md in each strategy directory!
