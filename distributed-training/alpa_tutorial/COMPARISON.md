# Alpa vs Manual Parallelism: Complete Comparison

This guide compares Alpa's automatic approach with manual parallelism strategies (ZeRO, PipeDream, Megatron).

---

## Quick Summary

| Aspect | Manual (ZeRO/PipeDream) | Alpa (Automatic) |
|--------|------------------------|------------------|
| **Setup Time** | Days to weeks | Minutes |
| **Code Complexity** | 100+ lines | 1 decorator |
| **Expertise Needed** | Expert-level | Beginner-friendly |
| **Performance** | Good (if tuned well) | Often better |
| **Flexibility** | Must retune for new models | Adapts automatically |
| **Framework** | PyTorch | JAX only |

---

## Detailed Comparison

### 1. Code Complexity

#### ZeRO (Manual Configuration)

```python
# Must choose strategy
CONFIG = {
    "strategy": "zero2",  # Or zero1? zero3? Which is best?
    "model": "gpt2-large",
    "batch_size": 4,      # Will this fit in memory?
}

# Must configure DeepSpeed
ds_config = {
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "train_batch_size": 16,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 1,
}

# Must launch with deepspeed
# deepspeed --num_gpus=4 train.py

# Total: ~50 lines of config + manual tuning
```

#### PipeDream (Manual Pipeline Definition)

```python
# Must decide pipeline stages
class PipelineModel(nn.Module):
    def __init__(self):
        # Which layers go in which stage?
        self.stage0 = nn.Sequential(layers[0:12])   # GPU 0
        self.stage1 = nn.Sequential(layers[12:24])  # GPU 1
        self.stage2 = nn.Sequential(layers[24:36])  # GPU 2
        self.stage3 = nn.Sequential(layers[36:48])  # GPU 3
    
    def forward(self, x):
        # Must handle microbatch splitting
        microbatches = split_into_microbatches(x, num_mb=4)
        
        # Must schedule forward/backward carefully
        for mb in microbatches:
            # Complex scheduling logic here...
            pass
        
        # Must implement weight versioning
        # Must handle gradient accumulation
        # Must synchronize across stages

# Total: ~200 lines of complex pipeline code
```

#### Alpa (Automatic)

```python
@parallelize
def train_step(model, batch):
    loss = model(batch)
    return loss

# That's it! Total: 1 line (the decorator)
```

---

### 2. Performance Tuning

#### Manual Approach

```
Week 1: Try ZeRO-2, batch_size=4
        → OOM error
        
Week 1: Try ZeRO-2, batch_size=2
        → Works but slow (60s/step)
        
Week 2: Try ZeRO-3, batch_size=2
        → Even slower (90s/step)
        
Week 2: Try ZeRO-2, batch_size=2, gradient_accumulation=2
        → Faster! (45s/step)
        
Week 3: Try pipeline_parallel=2, tensor_parallel=2
        → Setup complex, debugging...
        
Week 4: Finally found good config
        → 30s/step ✅
        
Total time: 4 weeks
```

#### Alpa Approach

```
Run 1: Alpa compiles (20 minutes)
       → Automatically finds optimal config
       → 28s/step ✅ (better than manual!)
       
Total time: 20 minutes
```

---

### 3. Configuration Space

#### Manual: Must Choose From Exponentially Many Options

For a 48-layer transformer on 8 GPUs:

**ZeRO choices:**
- Stage: 1, 2, or 3
- Batch size: 1, 2, 4, 8, ...
- Gradient accumulation: 1, 2, 4, 8, ...
- Overlapping: on/off
- Offload: on/off

**Pipeline choices:**
- Number of stages: 1, 2, 4, 8
- Layers per stage: many combinations
- Microbatches: 1, 2, 4, 8, 16, ...

**Tensor parallel choices:**
- TP degree: 1, 2, 4, 8
- Which layers to split: 2^48 combinations

**Total combinations:** > 1 million!

**Manual approach:** Try ~10-50 configs, hope you find a good one

**Alpa approach:** Searches intelligently, finds near-optimal automatically

---

### 4. When Model Changes

#### Manual Approach

```python
# Original model: GPT-2 (12 layers, 768 hidden)
CONFIG = {
    "pipeline_stages": 2,
    "tensor_parallel": 2,
    "microbatches": 4,
}
# This works great!

# New model: GPT-3 style (96 layers, 12288 hidden)
CONFIG = {
    "pipeline_stages": 2,  # ❌ Not optimal anymore!
    "tensor_parallel": 2,  # ❌ Should be 8!
    "microbatches": 4,     # ❌ Should be 16!
}
# Must retune everything from scratch (another 4 weeks)
```

#### Alpa Approach

```python
# Original model
@parallelize
def train(model, batch):
    return model(batch)
# Alpa finds optimal config

# New model (just change model definition)
@parallelize
def train(new_model, batch):
    return new_model(batch)
# Alpa automatically finds NEW optimal config!
# No retuning needed!
```

---

### 5. Communication Overhead

#### ZeRO-2 (Manual)

```
Per training step:
- AllGather gradients: 2× model size
- Reduce gradients: 2× model size
Total communication: 4× model size

For 7B model:
Communication per step: 28 GB
```

#### PipeDream (Manual)

```
Per training step:
- Forward activations: activation_size × num_stages
- Backward gradients: gradient_size × num_stages
- Pipeline bubbles: ~25% wasted time

For 7B model with 4 stages:
Communication: ~10 GB
BUT: 25% bubble time waste
```

#### Alpa (Automatic)

```
Alpa intelligently combines:
- Tensor parallelism where communication is cheap
- Pipeline parallelism where it reduces bubbles
- Data parallelism where model fits

Result:
Communication: ~8 GB
Bubble time: ~10%
✅ Best of both worlds!
```

---

### 6. Real-World Example: GPT-3 (175B)

#### Megatron-LM (Manual, by NVIDIA Experts)

```
Team: 10 expert engineers
Time: 3 months of tuning
Configuration:
- Data parallel: 8
- Pipeline parallel: 8
- Tensor parallel: 8
- Total GPUs: 512
- Microbatches: carefully tuned per layer

Performance: 140 TFLOPS/GPU (baseline)
```

#### Alpa (Automatic)

```
Team: 1 researcher
Time: 2 days (mostly compilation)
Configuration:
- Automatically determined all parallelism
- Total GPUs: 512

Performance: 155 TFLOPS/GPU (+10% faster!)
```

**Source:** Alpa paper, Table 2

---

### 7. Debugging Complexity

#### Manual Approach

```
Common issues:
1. Pipeline bubble too large → tune microbatches (1 week)
2. OOM error → tune batch size (3 days)
3. Slow communication → tune tensor parallel (1 week)
4. Load imbalance → rebalance pipeline stages (3 days)
5. Gradient explosion → tune gradient clipping (2 days)

Total debugging: 3-4 weeks typical
```

#### Alpa Approach

```
Common issues:
1. Compilation slow → wait (one-time cost)
2. OOM error → Alpa automatically reduces batch size
3. Slow → Alpa already optimized

Total debugging: Hours, not weeks
```

---

### 8. Framework Comparison

#### PyTorch (ZeRO, PipeDream, Megatron)

**Pros:**
- ✅ Most popular framework
- ✅ Large community
- ✅ Many pretrained models
- ✅ Good for production

**Cons:**
- ❌ Manual parallelization needed
- ❌ Complex APIs
- ❌ Hard to optimize

#### JAX (Alpa)

**Pros:**
- ✅ Automatic parallelization
- ✅ Clean functional design
- ✅ Great for research
- ✅ Faster compilation

**Cons:**
- ❌ Smaller community
- ❌ Fewer pretrained models
- ❌ Learning curve for functional style

---

## Decision Matrix

### Use Manual (ZeRO/PipeDream) If:

✅ You MUST use PyTorch  
✅ You have proven config that works  
✅ Your model doesn't change  
✅ You have expert team  
✅ Production system with strict requirements

### Use Alpa If:

✅ You want best performance  
✅ You're trying new architectures  
✅ You don't have time to tune  
✅ You're okay with JAX  
✅ Research or experimentation

---

## Performance Summary

Based on Alpa paper benchmarks:

| Model | Manual Best | Alpa | Speedup |
|-------|------------|------|---------|
| GPT-2 Medium (355M) | 100% | 105% | +5% |
| GPT-2 Large (774M) | 100% | 112% | +12% |
| GPT-3 Small (6.7B) | 100% | 108% | +8% |
| GPT-3 Medium (30B) | 100% | 110% | +10% |
| GPT-3 Large (175B) | 100% | 111% | +11% |

**Key insight:** Alpa consistently beats or matches best manual configs!

---

## Code Size Comparison

| Task | Manual Lines | Alpa Lines | Reduction |
|------|-------------|------------|-----------|
| Model definition | 200 | 200 | 0% |
| Parallelization | 300 | 1 | **99.7%** |
| Configuration | 100 | 0 | **100%** |
| **Total** | **600** | **201** | **66%** |

---

## Learning Curve

```
Manual Parallelism:
Week 1: Learn distributed basics
Week 2: Learn ZeRO concepts
Week 3: Learn pipeline parallelism  
Week 4: Learn tensor parallelism
Week 5-8: Practice and debug
Total: 2 months to proficiency

Alpa:
Day 1: Learn JAX basics (4 hours)
Day 2: Learn Flax (2 hours)
Day 3: Add @parallelize and run
Total: 1-2 days to proficiency
```

---

## Summary

**Manual parallelism (ZeRO, PipeDream):**
- More control
- PyTorch ecosystem
- Proven in production
- But: weeks of tuning, expert-level knowledge required

**Automatic parallelism (Alpa):**
- Minimal code changes
- Often better performance
- Adapts to model changes
- But: JAX-only, newer technology

**Recommendation:**
- **Learning?** Start with Alpa (easier, faster)
- **Production with PyTorch?** Use ZeRO
- **Research/Experimentation?** Use Alpa
- **New architecture?** Definitely Alpa

---

## Next Steps

1. Try both! Run ZeRO tutorial (`../README.md`)
2. Run Alpa tutorial (`README.md`)
3. Compare performance on same model
4. See which workflow you prefer

Both are valuable to know! 🚀
