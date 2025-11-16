# Tensor Parallelism: Deep Dive

## Introduction

Tensor Parallelism is Megatron-LM's core innovation. This document provides a mathematical and conceptual deep dive into how and why it works.

---

## The Problem: Matrix Multiplication at Scale

### Standard Forward Pass

In a transformer, the self-attention layer performs:

```
Y = X · W_qkv

Where:
- X: input tensor [B, S, H]
  - B = batch size
  - S = sequence length  
  - H = hidden dimension (e.g., 12,288 for GPT-3)
- W_qkv: weight matrix [H, 3H]
- Y: output [B, S, 3H] (queries, keys, values concatenated)

Memory required for W_qkv:
H × 3H × 4 bytes (FP32) = 12,288 × 36,864 × 4 = 1.8 GB
Just one matrix!

For GPT-3 with 96 layers:
96 layers × 8 matrices per layer × ~2 GB = ~1.5 TB
Doesn't fit on any GPU!
```

### Naive Solution: Split Randomly

❌ **Bad idea**: Split weight matrix arbitrarily

```
GPU 0: W_qkv[0:H/2, :]      → Partial result Y₀
GPU 1: W_qkv[H/2:H, :]      → Partial result Y₁

Problem:
- Y₀ and Y₁ are incomplete
- Need to communicate and combine
- Must synchronize after EVERY operation
- Communication overhead dominates!
```

---

## The Solution: Column and Row Parallelism

Megatron-LM uses a mathematically elegant split based on matrix multiplication properties.

### Mathematical Foundation

Key insight from linear algebra:

```
Given: Y = X · W

Column-wise split:
W = [W₁ | W₂]
Y = X · [W₁ | W₂] = [X·W₁ | X·W₂] = [Y₁ | Y₂]
Each GPU computes independently!

Row-wise split:
W = [W₁]
    [W₂]
Y = X · [W₁] = X·W₁ + X·W₂
         [W₂]
Need to sum results from both GPUs
```

### Application to Transformers

#### Step 1: Column Parallel (Q, K, V Computation)

```python
# Mathematical representation
X: [B, S, H]           # Input
W_qkv: [H, 3H]         # Weight matrix

# Split W_qkv by columns across N GPUs
GPU_i: W_qkv[:, i*3H/N : (i+1)*3H/N]

# Each GPU computes independently
GPU_0: Y₀ = X · W_qkv[:, 0:3H/N]        # Shape: [B, S, 3H/N]
GPU_1: Y₁ = X · W_qkv[:, 3H/N:6H/N]    # Shape: [B, S, 3H/N]
...
GPU_N: Y_N = X · W_qkv[:, 3H*(N-1)/N:] # Shape: [B, S, 3H/N]

# Concatenate results (no communication needed!)
Y = [Y₀ | Y₁ | ... | Y_N]              # Shape: [B, S, 3H]
```

**Why this works**:
- Each GPU gets a subset of attention heads
- Heads are independent by design
- No synchronization needed!

#### Step 2: Parallel Attention Computation

```python
# Split Q, K, V by attention heads
For GPU_i with heads h_i to h_{i+1}:

Q_i = Y₀[:, :, h_i:h_{i+1}]    # Local queries
K_i = Y₁[:, :, h_i:h_{i+1}]    # Local keys  
V_i = Y₂[:, :, h_i:h_{i+1}]    # Local values

# Compute attention independently
Attention_i = softmax(Q_i · K_iᵀ) · V_i

# Still no communication!
```

#### Step 3: Row Parallel (Output Projection)

```python
# Weight matrix for output
W_out: [H, H]

# Split W_out by ROWS across N GPUs
GPU_i: W_out[i*H/N : (i+1)*H/N, :]

# Each GPU computes partial result
GPU_0: Z₀ = Attention₀ · W_out[0:H/N, :]
GPU_1: Z₁ = Attention₁ · W_out[H/N:2H/N, :]
...

# Sum results (ALL-REDUCE required)
Z = Z₀ + Z₁ + ... + Z_N

⚠️ Communication Point #1
```

---

## Multi-Head Attention: Perfect for Parallelism

### Why Multi-Head Attention is Naturally Parallel

Standard transformer uses multi-head attention:

```
Number of heads: h (e.g., 96 for GPT-3)
Hidden dimension: H (e.g., 12,288)
Dimension per head: d = H / h (e.g., 128)

For each head i:
Q_i = X · W_q^i    where W_q^i: [H, d]
K_i = X · W_k^i    where W_k^i: [H, d]
V_i = X · W_v^i    where W_v^i: [H, d]

Attention_i = softmax(Q_i · K_iᵀ / √d) · V_i

Final output:
Y = Concat(Attention₁, Attention₂, ..., Attention_h) · W_out
```

### Head-Wise Parallelism

Distribute heads across GPUs:

```
96 total heads, 8 GPUs → 12 heads per GPU

GPU 0: Heads 0-11
├─ W_q for heads 0-11: [H, 12d]
├─ W_k for heads 0-11: [H, 12d]  
├─ W_v for heads 0-11: [H, 12d]
└─ Computes Attention_{0-11} independently

GPU 1: Heads 12-23
├─ W_q for heads 12-23: [H, 12d]
├─ W_k for heads 12-23: [H, 12d]
├─ W_v for heads 12-23: [H, 12d]
└─ Computes Attention_{12-23} independently

...

GPU 7: Heads 84-95
└─ Computes Attention_{84-95} independently

Concatenation is implicit (each GPU has its chunk)
Only need ALL-REDUCE for final output projection!
```

---

## Feed-Forward Network Parallelism

### Standard FFN

```python
# Two-layer MLP with GeLU activation
FFN(x) = W₂ · GeLU(W₁ · x)

Where:
W₁: [H, 4H]    # Expansion layer (12,288 → 49,152)
W₂: [4H, H]    # Projection layer (49,152 → 12,288)

Memory: (H × 4H + 4H × H) × 4 bytes
      = 2 × 12,288 × 49,152 × 4 bytes  
      = ~4.8 GB per layer
```

### Column-Parallel Expansion

```python
# Split W₁ by columns
GPU_i: W₁[:, i*4H/N : (i+1)*4H/N]

# Each GPU computes independently
GPU_0: Z₀ = GeLU(X · W₁[:, 0:4H/N])
GPU_1: Z₁ = GeLU(X · W₁[:, 4H/N:8H/N])
...
GPU_N: Z_N = GeLU(X · W₁[:, 4H*(N-1)/N:])

# No communication needed!
# Each GPU has 4H/N neurons
```

### Row-Parallel Projection

```python
# Split W₂ by rows
GPU_i: W₂[i*4H/N : (i+1)*4H/N, :]

# Each GPU computes partial result
GPU_0: Y₀ = Z₀ · W₂[0:4H/N, :]
GPU_1: Y₁ = Z₁ · W₂[4H/N:8H/N, :]
...

# Sum results (ALL-REDUCE required)
Y = Y₀ + Y₁ + ... + Y_N

⚠️ Communication Point #2
```

---

## Communication Analysis

### Communication Volume

For a transformer layer with:
- Batch size: B
- Sequence length: S
- Hidden dimension: H
- Tensor parallel degree: N

#### Per-Layer Communication

```
Forward Pass:
├─ Attention output: ALL-REDUCE of [B, S, H]
│  Volume = B × S × H × 4 bytes (FP32)
│
└─ FFN output: ALL-REDUCE of [B, S, H]  
   Volume = B × S × H × 4 bytes (FP32)

Total forward: 2 × B × S × H × 4 bytes

Backward Pass:
├─ Gradient w.r.t attention input: ALL-REDUCE of [B, S, H]
└─ Gradient w.r.t FFN input: ALL-REDUCE of [B, S, H]

Total backward: 2 × B × S × H × 4 bytes

Per-layer total: 4 × B × S × H × 4 bytes
```

#### Example: GPT-3

```
B = 512 (batch size)
S = 2048 (sequence length)
H = 12,288 (hidden dimension)
Layers = 96

Per-layer communication:
4 × 512 × 2048 × 12,288 × 4 bytes = ~200 MB

Total per training step:
96 layers × 200 MB = ~19.2 GB

On NVLink (600 GB/s):
Communication time: 19.2 GB / 600 GB/s = 32 ms

Computation time: ~500 ms per step

Communication overhead: 32/500 = 6.4%
```

### ALL-REDUCE Implementation

Efficient ALL-REDUCE using Ring Algorithm:

```
For N GPUs in a ring:

Step 1: Scatter-Reduce
GPU 0 → GPU 1 → GPU 2 → ... → GPU N → GPU 0
Each GPU receives and accumulates its chunk
Steps: N-1

Step 2: All-Gather  
GPU 0 → GPU 1 → GPU 2 → ... → GPU N → GPU 0
Each GPU forwards complete chunk
Steps: N-1

Total steps: 2(N-1)
Data transferred per GPU: 2(N-1)/N × Data size
Approaches: 2 × Data size as N increases

For N=8: Each GPU transfers ~1.75× data size
```

---

## Memory Savings

### Parameter Memory

```
Single GPU (No parallelism):
W_qkv: [H, 3H] = H × 3H parameters
W_out: [H, H]  = H × H parameters  
W₁:    [H, 4H] = H × 4H parameters
W₂:    [4H, H] = 4H × H parameters

Total per layer: H × 3H + H × H + H × 4H + 4H × H
                = H × (3H + H + 4H + 4H)
                = 12H² parameters

For H = 12,288:
12 × 12,288² × 4 bytes = ~7.2 GB per layer

96 layers: ~691 GB just for parameters!
```

```
With Tensor Parallelism (N=8 GPUs):
Each GPU holds:
W_qkv: [H, 3H/8]
W_out: [H/8, H]
W₁:    [H, 4H/8]  
W₂:    [4H/8, H]

Total per GPU: 12H² / 8 parameters

For H = 12,288:
~86 GB per layer per GPU

96 layers: ~8.6 GB per GPU ✅ Manageable!
```

### Activation Memory

Activations also split across GPUs:

```
Without Tensor Parallel:
Attention output: [B, S, H]
FFN intermediate: [B, S, 4H]

Per layer: B × S × (H + 4H) = 5BSH

With Tensor Parallel (N=8):
Attention output: [B, S, H/8] per GPU
FFN intermediate: [B, S, 4H/8] per GPU

Per layer per GPU: 5BSH / 8

For B=512, S=2048, H=12,288:
Without: ~200 MB per layer
With (N=8): ~25 MB per layer per GPU
```

---

## Gradient Flow and Backpropagation

### Column-Parallel Backward

```python
Forward (Column-Parallel):
Y = X · W    where W = [W₁ | W₂ | ... | W_N]
Y = [Y₁ | Y₂ | ... | Y_N]

Backward:
∂L/∂X = ∂L/∂Y · Wᵀ

Since Y = [Y₁ | Y₂ | ... | Y_N] and W = [W₁ | W₂ | ... | W_N]:
∂L/∂Y = [∂L/∂Y₁ | ∂L/∂Y₂ | ... | ∂L/∂Y_N]

∂L/∂X = ∂L/∂Y₁·W₁ᵀ + ∂L/∂Y₂·W₂ᵀ + ... + ∂L/∂Y_N·W_Nᵀ

⚠️ Requires ALL-REDUCE to sum gradients!
```

### Row-Parallel Backward

```python
Forward (Row-Parallel):
Y = X · W    where W = [W₁]
                       [W₂]
                       [...]
                       [W_N]
Y = X₁·W₁ + X₂·W₂ + ... + X_N·W_N  (already summed via ALL-REDUCE)

Backward:
∂L/∂X = ∂L/∂Y · Wᵀ

Since W is split by rows:
∂L/∂X_i = ∂L/∂Y · W_iᵀ

✅ No ALL-REDUCE needed! Gradient naturally splits.
```

### The Symmetry

Beautiful mathematical symmetry:

```
Column-Parallel:
├─ Forward: No ALL-REDUCE
└─ Backward: Needs ALL-REDUCE

Row-Parallel:
├─ Forward: Needs ALL-REDUCE  
└─ Backward: No ALL-REDUCE

Pairing them alternately:
Column → Row → Column → Row
Minimizes total communication!
```

---

## Practical Considerations

### Load Balancing

Ensure even distribution:

```
Number of heads = 96
Tensor parallel degree = 8
Heads per GPU = 96 / 8 = 12 ✅ Even split

If heads = 100 and N = 8:
100 / 8 = 12.5 ❌ Not evenly divisible
Need to pad or adjust configuration
```

### Numerical Precision

Communication in lower precision:

```
Forward/Backward activations: FP16
├─ 2× less bandwidth
├─ 2× faster communication
└─ Minimal accuracy impact

Model parameters: FP32
├─ Stored in FP32
├─ Converted to FP16 for forward pass
└─ Gradients accumulated in FP32
```

### Sequence Parallelism (Advanced)

For very long sequences, also split sequence dimension:

```
Standard: Each GPU processes full sequence [B, S, H/N]
Sequence Parallel: Each GPU processes [B, S/M, H/N]

Benefits:
├─ Further memory reduction
├─ Supports longer sequences
└─ Additional communication dimension

Tradeoff:
├─ More complex implementation
└─ More communication points
```

---

## Summary

### Key Mathematical Insights

1. **Column-parallel exploits independence**
   - Multi-head attention heads are independent
   - FFN neurons are independent
   - No communication in forward pass

2. **Row-parallel requires synchronization**
   - Partial results must be summed
   - ALL-REDUCE is necessary
   - Backward pass is free

3. **Alternating column/row minimizes communication**
   - Only 2 ALL-REDUCE per layer
   - Symmetric forward/backward
   - Optimal for transformers

### Why Tensor Parallelism Works

```
✅ Exploits transformer structure
   (Multi-head attention is naturally parallel)

✅ Minimizes communication
   (Only 2 synchronization points per layer)

✅ Balances computation
   (Even distribution of heads/neurons)

✅ Scales efficiently
   (Communication O(BSH), not O(parameters))

✅ Mathematical elegance
   (Column/row split symmetry)
```

### The Bottom Line

Tensor parallelism is not just "splitting the model" - it's a **mathematically principled** approach that exploits the structure of transformers to achieve near-perfect parallelization with minimal communication.

This is why Megatron-LM can train 175B+ parameter models efficiently!
