# 3D Parallelism: Combining All Dimensions

## Introduction

**3D Parallelism** is the combination of three orthogonal parallelization strategies:
1. **Data Parallelism** (D) - Across data batches
2. **Pipeline Parallelism** (P) - Across model layers
3. **Tensor Parallelism** (T) - Within individual layers

This document explains how they work together to enable training of trillion-parameter models.

---

## The Three Dimensions

### Dimension 1: Data Parallelism (D)

**What it does**: Replicates the entire model and processes different data batches

```
Original batch size: 512
Data parallel degree: D = 4
Micro-batch per GPU: 512 / 4 = 128

GPU Group 0: Processes samples 0-127
GPU Group 1: Processes samples 128-255
GPU Group 2: Processes samples 256-383
GPU Group 3: Processes samples 384-511

After forward/backward:
└─ ALL-REDUCE gradients across all groups
└─ All groups update with same gradients
```

**When to use**:
- Model fits in GPU memory (with pipeline/tensor parallel)
- Want to increase effective batch size
- Have multiple copies of the model

### Dimension 2: Pipeline Parallelism (P)

**What it does**: Splits model layers into sequential stages

```
96-layer model, P = 4 stages:

Stage 0 (GPU 0-N):  Layers 0-23
Stage 1 (GPU N-2N): Layers 24-47
Stage 2 (GPU 2N-3N): Layers 48-71
Stage 3 (GPU 3N-4N): Layers 72-95

Data flows: Stage 0 → Stage 1 → Stage 2 → Stage 3
```

**When to use**:
- Model depth doesn't fit in memory
- Have sequential layer structure
- Can tolerate pipeline bubbles

### Dimension 3: Tensor Parallelism (T)

**What it does**: Splits individual layers across GPUs

```
Single attention layer, T = 8:

GPU 0: Attention heads 0-11
GPU 1: Attention heads 12-23
GPU 2: Attention heads 24-35
...
GPU 7: Attention heads 84-95

All GPUs compute in parallel for one layer
```

**When to use**:
- Individual layers don't fit in memory
- Model width is very large
- Have high-bandwidth interconnect (NVLink)

---

## How They Combine: The 3D Cube

### Visualization

```
                Pipeline Dimension (P=4)
                      ↓
            ┌──────────────────────┐
           /│  S0    S1    S2    S3│
          / │ L0-23 L24-47 L48-71 L72-95
         /  │                      │
        /   │      Each stage has  │
       /    │      T=8 GPUs for   │
      /     │      tensor parallel │
     /      │                      │
    /       └──────────────────────┘
   /       /                      /
  /       /  Tensor Parallel     /
 /       /   Dimension (T=8)    /
/       /         ↓            /
└──────┴──────────────────────┘
│  This entire cube is         │
│  replicated D=2 times for    │
│  data parallelism            │
└─────────────────────────────→
    Data Parallel Dimension (D=2)

Total GPUs = D × P × T = 2 × 4 × 8 = 64
```

### GPU Assignment

```
For D=2, P=4, T=8 (64 total GPUs):

Data Replica 0:
├─ Pipeline Stage 0: GPUs 0-7   (tensor parallel)
├─ Pipeline Stage 1: GPUs 8-15  (tensor parallel)
├─ Pipeline Stage 2: GPUs 16-23 (tensor parallel)
└─ Pipeline Stage 3: GPUs 24-31 (tensor parallel)

Data Replica 1:
├─ Pipeline Stage 0: GPUs 32-39 (tensor parallel)
├─ Pipeline Stage 1: GPUs 40-47 (tensor parallel)
├─ Pipeline Stage 2: GPUs 48-55 (tensor parallel)
└─ Pipeline Stage 3: GPUs 56-63 (tensor parallel)
```

---

## Memory Distribution

### Example: GPT-3 175B on 1024 GPUs

Configuration: D=8, P=16, T=8

```
Total Parameters: 175 billion
Parameter Memory: 175B × 4 bytes (FP32) = 700 GB

Distribution:
├─ Pipeline splits 96 layers into 16 stages
│  └─ Each stage: 6 layers
│
├─ Tensor parallel splits each layer across 8 GPUs
│  └─ Parameters per GPU: 700 GB / (16 × 8) = ~5.5 GB
│
└─ Data parallel replicates 8 times
   └─ No additional memory (same parameters)

Memory per GPU breakdown:
├─ Model parameters: ~5.5 GB
├─ Gradients: ~5.5 GB
├─ Optimizer states (Adam): ~16.5 GB (3× parameters)
├─ Activations: ~10 GB (depends on micro-batch size)
├─ Working memory: ~2.5 GB
└─ Total: ~40 GB per GPU ✅ Fits on A100 (80GB)
```

### Memory Scaling Law

```
For model with M parameters on D×P×T GPUs:

Memory per GPU ≈ (M × 4 bytes) / (P × T)
                 + activations
                 + optimizer overhead

Pipeline (P): Divides layers
Tensor (T): Divides layer width
Data (D): Doesn't affect memory (replicates model)
```

---

## Communication Patterns

### Three Independent Communication Groups

```
1. Tensor Parallel Group (T=8):
   ┌────────────────────────────┐
   │ GPU 0 ↔ GPU 1 ↔ ... ↔ GPU 7│
   └────────────────────────────┘
   Communication: ALL-REDUCE
   Frequency: 2× per layer
   Volume: O(batch_size × seq_len × hidden)
   Requirement: NVLink (high bandwidth, low latency)

2. Pipeline Parallel Group (P=4):
   GPU Set 0 → GPU Set 1 → GPU Set 2 → GPU Set 3
   Communication: POINT-TO-POINT (send/recv)
   Frequency: Per microbatch
   Volume: O(batch_size × seq_len × hidden)
   Requirement: InfiniBand (good bandwidth)

3. Data Parallel Group (D=2):
   Replica 0 GPUs ↔ Replica 1 GPUs
   Communication: ALL-REDUCE (gradients)
   Frequency: Once per training step
   Volume: O(model_parameters / (P × T))
   Requirement: InfiniBand (can overlap with computation)
```

### Communication Hierarchy

```
Most Frequent ────────────────────→ Least Frequent
Highest BW Req ───────────────────→ Lowest BW Req

Tensor Parallel    Pipeline Parallel    Data Parallel
     (T)                 (P)                 (D)
      ↓                   ↓                   ↓
   NVLink            InfiniBand          InfiniBand
   2×/layer          per microbatch      per step
   ~5-10% overhead   ~10-15% overhead    ~3-5% overhead
```

---

## Training Step Execution

### Forward Pass

```
For microbatch m:

1. Data Parallel: Each replica processes different data
   Replica 0: microbatch m₀
   Replica 1: microbatch m₁
   ...

2. Pipeline Parallel: Flow through stages
   Stage 0 processes m₀ → sends to Stage 1
   Stage 1 processes m₀ → sends to Stage 2
   Stage 2 processes m₀ → sends to Stage 3
   Stage 3 processes m₀ → computes loss

3. Tensor Parallel: Within each stage
   For each layer in stage:
   ├─ Split computation across T GPUs
   ├─ ALL-REDUCE after attention
   └─ ALL-REDUCE after FFN

Timeline (Pipeline stages working on different microbatches):
Time →
────────────────────────────────────────────────────
Stage 0: [m₀] [m₁] [m₂] [m₃] ...
Stage 1:      [m₀] [m₁] [m₂] [m₃] ...
Stage 2:           [m₀] [m₁] [m₂] [m₃] ...
Stage 3:                [m₀] [m₁] [m₂] [m₃] ...
────────────────────────────────────────────────────
         ↑ Pipeline bubble (idle time)
```

### Backward Pass

```
1. Gradients flow backward through pipeline
   Stage 3 receives loss → computes grads → sends to Stage 2
   Stage 2 receives grads → computes grads → sends to Stage 1
   Stage 1 receives grads → computes grads → sends to Stage 0

2. Tensor parallel gradients
   Within each stage, for each layer:
   ├─ Compute local gradients
   ├─ ALL-REDUCE to synchronize
   └─ Each GPU has full gradient for its parameters

3. Data parallel gradient sync
   After all microbatches processed:
   └─ ALL-REDUCE gradients across data parallel replicas
```

### Weight Update

```
After all microbatches and gradient sync:

1. Each GPU has gradients for its subset of parameters
   (Due to pipeline and tensor parallelism split)

2. Apply optimizer update locally
   GPU_i updates its own parameters
   No communication needed!

3. Next iteration starts with updated weights
```

---

## Configuration Guidelines

### Choosing Dimensions

```
Total GPUs available: N
Model parameters: M
Max layer size: L
Memory per GPU: G

1. Choose Tensor Parallel (T):
   ├─ If single layer fits: T = 1
   ├─ If layer needs split: T = 2, 4, or 8
   └─ Constraint: Must have NVLink within T-group
   
   Rule: L / T < G (layer fits after T-way split)

2. Choose Pipeline Parallel (P):
   ├─ If model fits after T-split: P = 1
   ├─ If need more splitting: P = 2, 4, 8, 16, ...
   └─ Constraint: P should divide num_layers evenly
   
   Rule: M / (T × P) < G (model fits with T and P)

3. Choose Data Parallel (D):
   ├─ Use remaining GPUs: D = N / (T × P)
   └─ Higher D = larger effective batch size
   
   Rule: D = N / (T × P)
```

### Example Configurations

#### Small Model (1.3B parameters, 24 layers)

```
GPUs available: 8
Layers: 24
Hidden: 2048

Configuration:
├─ T = 1 (layers fit on single GPU)
├─ P = 1 (model fits in memory)
└─ D = 8 (use all GPUs for data parallel)

Result: Simple data parallelism
```

#### Medium Model (13B parameters, 40 layers)

```
GPUs available: 64
Layers: 40
Hidden: 5120

Configuration:
├─ T = 4 (split wide layers)
├─ P = 4 (split into 4 stages of 10 layers)
└─ D = 4 (64 / (4 × 4) = 4)

Result: Balanced 3D parallelism
```

#### Large Model (175B parameters, 96 layers)

```
GPUs available: 1024
Layers: 96
Hidden: 12,288

Configuration:
├─ T = 8 (layers very wide, need splitting)
├─ P = 16 (96 layers → 6 layers per stage)
└─ D = 8 (1024 / (8 × 16) = 8)

Result: Full 3D parallelism (GPT-3 configuration)
```

#### Extreme Model (1T parameters, 128 layers)

```
GPUs available: 4096
Layers: 128
Hidden: 25,600

Configuration:
├─ T = 16 (extremely wide layers)
├─ P = 32 (128 layers → 4 layers per stage)
└─ D = 8 (4096 / (16 × 32) = 8)

Result: Maximum parallelization
```

---

## Performance Analysis

### Efficiency Factors

```
Ideal Throughput = GPUs × GPU_FLOPS × Utilization

Real Throughput = Ideal × E_tensor × E_pipeline × E_data

Where:
├─ E_tensor: Tensor parallel efficiency
│  └─ Reduced by: ALL-REDUCE overhead
│  └─ Typical: 85-95%
│
├─ E_pipeline: Pipeline parallel efficiency
│  └─ Reduced by: Pipeline bubbles
│  └─ Typical: 80-90%
│
└─ E_data: Data parallel efficiency
   └─ Reduced by: Gradient sync
   └─ Typical: 95-98%

Overall: 0.85 × 0.85 × 0.95 ≈ 69% of peak
```

### Pipeline Bubble Analysis

Pipeline efficiency depends on number of microbatches:

```
Pipeline stages: P
Microbatches: M

Ideal time: M × time_per_microbatch
Actual time: (M + P - 1) × time_per_microbatch

Bubble overhead: (P - 1) / (M + P - 1)

Examples:
├─ P=4, M=8:  (4-1)/(8+4-1) = 27% bubble
├─ P=4, M=16: (4-1)/(16+4-1) = 16% bubble
├─ P=4, M=32: (4-1)/(32+4-1) = 9% bubble

Rule: Use M ≥ 4×P to keep bubbles < 10%
```

### Memory vs Efficiency Tradeoff

```
More microbatches:
✅ Better pipeline efficiency
❌ More activation memory

Fewer microbatches:
✅ Less memory needed
❌ Worse pipeline efficiency

Sweet spot: M = 4×P to 8×P
```

---

## Advanced Techniques

### Interleaved Pipeline Scheduling

Instead of assigning consecutive layers to stages, interleave them:

```
Standard: Each stage has consecutive layers
├─ Stage 0: Layers 0-23
├─ Stage 1: Layers 24-47
├─ Stage 2: Layers 48-71
└─ Stage 3: Layers 72-95

Interleaved: Each stage has spread-out layers
├─ Stage 0: Layers 0, 4, 8, 12, ..., 92
├─ Stage 1: Layers 1, 5, 9, 13, ..., 93
├─ Stage 2: Layers 2, 6, 10, 14, ..., 94
└─ Stage 3: Layers 3, 7, 11, 15, ..., 95

Benefits:
├─ Reduces pipeline bubble
├─ Better load balancing
└─ Can reduce bubble by ~50%
```

### Sequence Parallelism

Split sequence dimension in addition to tensor parallelism:

```
Standard Tensor Parallel:
Input: [batch, sequence, hidden/T]

With Sequence Parallel:
Input: [batch, sequence/S, hidden/T]

Benefits:
├─ Supports longer sequences
├─ Reduces memory for activations
└─ Better for long-context models

Tradeoff:
├─ Additional communication
└─ More complex implementation
```

### Selective Activation Recomputation

Recompute activations instead of storing them:

```
Without recomputation:
├─ Store all activations: High memory
└─ Fast backward pass

With full recomputation:
├─ Store minimal activations: Low memory
└─ Slow backward pass (recompute everything)

Selective (Megatron-LM approach):
├─ Store: Attention scores, layer outputs
├─ Recompute: QKV projections, FFN intermediate
└─ Balance: ~30% memory reduction, ~15% slowdown
```

---

## Best Practices

### DO:

✅ **Use T=8 for large transformers**
   - Optimal for NVLink topology (8 GPUs per node)
   - Good balance of parallelism and efficiency

✅ **Set P to divide layers evenly**
   - 96 layers → P = 2, 4, 6, 8, 12, 16, 24, 32, 48
   - Uneven splits cause load imbalance

✅ **Use M ≥ 4×P microbatches**
   - Keeps pipeline bubbles < 10%
   - Good memory/efficiency tradeoff

✅ **Maximize D within memory constraints**
   - Higher effective batch size
   - Better training stability

✅ **Profile and tune**
   - Measure actual throughput
   - Adjust based on your hardware

### DON'T:

❌ **Use tensor parallel across nodes**
   - Requires ultra-low latency (NVLink)
   - InfiniBand too slow

❌ **Make P too large**
   - Pipeline bubbles dominate
   - Need many microbatches

❌ **Use T=1 for huge layers**
   - Layers won't fit in memory
   - Underutilizes GPUs

❌ **Ignore pipeline balance**
   - Unequal stage times cause bubbles
   - Slower stages bottleneck whole pipeline

❌ **Forget about activation memory**
   - Can exceed parameter memory
   - Need selective recomputation

---

## Summary

### The Power of 3D Parallelism

```
┌─────────────────────────────────────────────┐
│  WHY 3D PARALLELISM ENABLES SCALE           │
├─────────────────────────────────────────────┤
│                                             │
│  Tensor Parallel (T):                       │
│  └─ Splits wide layers across GPUs          │
│                                             │
│  Pipeline Parallel (P):                     │
│  └─ Splits deep models across GPUs          │
│                                             │
│  Data Parallel (D):                         │
│  └─ Increases batch size and throughput     │
│                                             │
│  Together:                                  │
│  ├─ Trains trillion-parameter models        │
│  ├─ Scales to thousands of GPUs             │
│  ├─ Achieves 70-90% efficiency              │
│  └─ Makes the impossible possible!          │
│                                             │
└─────────────────────────────────────────────┘
```

### Key Takeaways

1. **Orthogonal dimensions** - Each handles different constraint
2. **Independent communication** - Different groups, different patterns
3. **Multiplicative scaling** - Total GPUs = D × P × T
4. **Configuration matters** - Wrong setup kills performance
5. **Hardware awareness** - NVLink for T, InfiniBand for P/D

### The Formula for Success

```
Training Large Models:
1. Choose T based on layer width
2. Choose P based on model depth  
3. Choose D to use remaining GPUs
4. Tune microbatches for efficiency
5. Profile and adjust
```

This is how GPT-3 and beyond became possible! 🚀
