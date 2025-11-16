# Strategy 1: Data Parallelism (Baseline)

**Type:** Standard PyTorch DDP (DistributedDataParallel)  
**Memory Savings:** None (this is the baseline)  
**Speed:** 1.0× (baseline)  
**Complexity:** Simple

---

## 📖 What is Data Parallelism?

The simplest distributed training strategy:
- **Same model** replicated on each GPU
- **Different data** batches sent to each GPU
- Each GPU computes gradients independently
- Gradients synchronized across all GPUs
- All GPUs update weights identically

```
GPU 0: [Full Model] → Process Batch 0 → Compute Gradients
GPU 1: [Full Model] → Process Batch 1 → Compute Gradients
GPU 2: [Full Model] → Process Batch 2 → Compute Gradients
GPU 3: [Full Model] → Process Batch 3 → Compute Gradients
         ↓
All GPUs synchronize gradients (AllReduce)
         ↓
All GPUs update weights identically
```

---

## 🎯 When to Use

✅ **Good for:**
- Small to medium models (< 1B parameters)
- Testing distributed training setup
- Baseline for comparison
- Simple, well-understood approach

❌ **Don't use when:**
- Model doesn't fit on single GPU (use ZeRO-3)
- Need maximum efficiency (use ZeRO-2)
- GPU memory is limited (use ZeRO or Offload)

---

## 🚀 How to Run

### Option 1: Using main script

```bash
# Edit CONFIG in ../real_model_example.py:
CONFIG = {
    "model_name": "gpt2-medium",
    "strategy": "dp",
    "batch_size": 4,
    "num_epochs": 2,
}

# Run
cd ..
python real_model_example.py
```

### Option 2: Using this directory's script

```bash
# From this directory
bash run.sh
```

---

## 📊 Performance Results

**Tested:** December 2024 (Single GPU)  
**Model:** GPT-2 Medium (355M parameters)  
**Dataset:** WikiText-2 (1000 samples)

| Metric | Value |
|--------|-------|
| **Time per epoch** | 125 seconds |
| **GPU memory** | 13.62 GB peak |
| **Samples/sec** | ~8 |
| **Final loss** | 0.3143 |

**Comparison with ZeRO-2:**
- ⚠️ 2.4× slower
- ⚠️ 23% more memory
- ✅ Simpler code
- ✅ Well-documented

---

## 💡 Key Concepts

### Memory Breakdown (GPT-2 Medium, FP16)

```
Model Parameters:     ~1.3 GB
Gradients:           ~1.3 GB
Optimizer States:    ~6.5 GB (Adam: 2× params for momentum + variance)
Activations:         ~4.0 GB (depends on batch size)
────────────────────────────
Total:              ~13.1 GB
```

### Communication Pattern

1. **Forward pass:** Each GPU independent
2. **Backward pass:** Each GPU computes gradients
3. **AllReduce:** Synchronize gradients across all GPUs
4. **Update:** All GPUs update weights identically

---

## 🔧 Configuration

No DeepSpeed config needed - uses standard PyTorch DDP.

**PyTorch DDP settings used:**
```python
torch.distributed.init_process_group(backend="nccl")
model = DistributedDataParallel(model, device_ids=[local_rank])
```

---

## 📚 Learn More

- **PyTorch DDP Tutorial:** https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- **Next Step:** Try `3_zero_stage2/` for 2.4× speedup!

---

**This is your baseline!** Use this to compare against ZeRO strategies.
