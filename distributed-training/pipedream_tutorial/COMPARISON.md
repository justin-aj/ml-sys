# PipeDream vs ZeRO: When to Use Which?

This guide helps you choose between Pipeline Parallelism (PipeDream) and Model Parallelism (ZeRO).

---

## Quick Comparison

| Feature | PipeDream (Pipeline) | ZeRO (Model Parallel) |
|---------|---------------------|----------------------|
| **Model Split** | Layers across GPUs | Weights/gradients across GPUs |
| **Data Flow** | Sequential (layer by layer) | Parallel (all GPUs compute together) |
| **Communication** | Activations between stages | Gradients across all GPUs |
| **GPU Utilization** | ~75% (with microbatches) | ~95% (all GPUs compute) |
| **Memory Savings** | Split model size | Optimizer states + gradients |
| **Best For** | Very deep models | Wide models with many parameters |

---

## When to Use PipeDream

✅ **Use PipeDream when:**
1. Your model has **many layers** (e.g., 100+ layers)
2. The model **doesn't fit on one GPU** even with batch size 1
3. Layers can be naturally **partitioned** (e.g., transformer layers)
4. You have **fast inter-GPU communication** (NVLink)

**Example:** GPT-3 with 96 layers
```python
# Split across 8 GPUs
GPU0: Layers 1-12
GPU1: Layers 13-24
GPU2: Layers 25-36
...
GPU7: Layers 85-96
```

❌ **Don't use PipeDream when:**
- Your model fits on one GPU (use Data Parallelism instead)
- You need maximum throughput (pipeline has some bubble time)
- Model has complex skip connections (hard to partition)

---

## When to Use ZeRO

✅ **Use ZeRO when:**
1. Your model **fits on one GPU**, but optimizer states don't
2. You want to train **larger batch sizes**
3. You want **minimal code changes** (drop-in replacement for DDP)
4. You have **multiple GPUs available**

**Example:** GPT-2 Medium (355M params)
```python
# ZeRO-2: Split optimizer and gradients across 4 GPUs
- 2.4× faster than Data Parallelism
- 23% less memory per GPU
- Same model on all GPUs!
```

❌ **Don't use ZeRO when:**
- Model is SO large it doesn't fit even when sharded (use Pipeline)
- You only have 1 GPU (ZeRO needs multiple GPUs)

---

## Real-World Examples

### Example 1: GPT-2 Medium (355M params)

**Scenario:** You have 4 GPUs, want to train GPT-2 Medium

**Recommendation:** ✅ Use ZeRO-2
- **Why:** Model fits on 1 GPU, ZeRO gives 2.4× speedup
- **Results:** 52s/epoch vs 125s/epoch (Data Parallel)
- **Memory:** 10.55 GB vs 13.62 GB

**Setup:**
```python
# From distributed-training/ tutorial
CONFIG = {
    "model_name": "gpt2-medium",
    "strategy": "zero2",
    "batch_size": 4,
    "num_epochs": 3,
}
```

### Example 2: GPT-3 (175B params)

**Scenario:** You have 64 GPUs, want to train GPT-3

**Recommendation:** ✅ Use PipeDream + ZeRO together!
- **Why:** Model too large for any single GPU
- **Setup:** 
  - Pipeline parallelism across 8 nodes (8 stages)
  - ZeRO-3 within each stage (8 GPUs per stage)
  - Total: 64 GPUs

**Hybrid approach:**
```
Node 0 (Layers 1-12):   8 GPUs with ZeRO-3
Node 1 (Layers 13-24):  8 GPUs with ZeRO-3
...
Node 7 (Layers 85-96):  8 GPUs with ZeRO-3
```

### Example 3: ResNet-50 (25M params)

**Scenario:** You have 4 GPUs, want to train ResNet-50

**Recommendation:** ✅ Use Data Parallelism (standard DDP)
- **Why:** Model easily fits on 1 GPU
- **Don't need:** ZeRO or Pipeline (overkill for small model)

---

## Decision Tree

```
Start: Do you have multiple GPUs?
│
├─ No (1 GPU)
│  └─ Use standard training (no distributed)
│
└─ Yes (2+ GPUs)
   │
   ├─ Does model fit on 1 GPU?
   │  │
   │  ├─ Yes
   │  │  └─ Model < 100M params?
   │  │     ├─ Yes → Use Data Parallelism (DDP)
   │  │     └─ No → Use ZeRO-2 (faster + less memory)
   │  │
   │  └─ No
   │     └─ Use Pipeline Parallelism (PipeDream)
   │        ├─ Want even more savings? → Add ZeRO-3 within stages
   │        └─ Have NVMe? → Use ZeRO-Infinity
```

---

## Performance Comparison (Tested)

From our tutorials on single GPU:

| Model | Strategy | Memory | Time/Epoch | Speedup |
|-------|----------|--------|------------|---------|
| gpt2-medium | Data Parallel | 13.62 GB | 125s | 1.0× |
| gpt2-medium | ZeRO-1 | 9.64 GB | 52s | 2.4× ✅ |
| gpt2-medium | ZeRO-2 | 10.55 GB | 52s | 2.4× ✅ |
| gpt2-large | ZeRO-3 | 21.03 GB | 105s | Larger model! |
| gpt2-large | ZeRO-Offload | 10.10 GB | 375s | 52% less GPU |

**Key insight:** ZeRO-2 is fastest for models that fit on GPU!

---

## Combining Strategies

You can combine multiple parallelism strategies:

### 3D Parallelism (State-of-the-art)

```
Data Parallelism (across nodes)
  └─ Pipeline Parallelism (across GPUs in node)
      └─ Model Parallelism (ZeRO within pipeline stage)
```

**Used by:** GPT-3, Megatron-LM, DeepSpeed

**Example with 128 GPUs:**
- 4 data parallel replicas (4 copies of model)
- 8 pipeline stages per replica
- 4 GPUs per stage with ZeRO-3
- Total: 4 × 8 × 4 = 128 GPUs

---

## Code Examples

### ZeRO-2 (from distributed-training/)
```python
CONFIG = {
    "model_name": "gpt2-medium",
    "strategy": "zero2",
    "batch_size": 4,
}
deepspeed --num_gpus=4 real_model_example.py
```

### Pipeline Parallelism (from pipedream_tutorial/)
```python
CONFIG = {
    "num_stages": 4,
    "num_microbatches": 4,
    "layers_per_stage": 12,
}
python pipedream_simple.py
```

### Hybrid (conceptual)
```python
# Use DeepSpeed with pipeline + ZeRO
ds_config = {
    "train_batch_size": 32,
    "zero_optimization": {"stage": 3},
    "pipeline": {
        "stages": 4,
        "micro_batches": 4
    }
}
```

---

## Summary

**For most users:**
- Small models (< 1B params): Use Data Parallelism
- Medium models (1-10B params): Use ZeRO-2 or ZeRO-3
- Large models (10B+ params): Use Pipeline + ZeRO hybrid

**Our tutorials:**
- `distributed-training/` → Learn ZeRO (start here!)
- `pipedream_tutorial/` → Learn Pipeline Parallelism

**Next steps:**
1. Run both tutorials
2. Understand the trade-offs
3. Choose based on your model size
4. Scale up to multi-GPU/multi-node

---

**Questions?** Check out:
- `distributed-training/README.md` - ZeRO tutorial
- `pipedream_tutorial/README.md` - Pipeline tutorial
- `distributed-training/TRAINING_RESULTS.md` - Real performance data

🚀 **Happy training!**
