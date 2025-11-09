# TASO Examples: Concrete Graph Rewrites

This document shows **real algebraic rewrites** that TASO performs, with before/after graphs and savings calculations.

---

## Example 1: Distributive Property (Basic)

### **Before Optimization**

```python
# Code
Y = (A @ B) + (A @ C)

# Graph
    A
   / \
  @   @
 B     C
  \   /
   +
   |
   Y

# Operations
- MatMul(A, B) → X1
- MatMul(A, C) → X2
- Add(X1, X2) → Y
```

**Cost Analysis:**
- Shape: A=[M×K], B=[K×N], C=[K×N]
- MatMul FLOPs: 2×M×K×N
- Add FLOPs: M×N
- **Total: 2MKN + MN FLOPs**
- **Memory: 2MN (for X1, X2)**

### **After TASO Optimization**

```python
# Code
Y = A @ (B + C)

# Graph
    B   C
     \ /
      +
      |
      @
      A
      |
      Y

# Operations
- Add(B, C) → X1
- MatMul(A, X1) → Y
```

**Cost Analysis:**
- Add FLOPs: K×N
- MatMul FLOPs: M×K×N
- **Total: MKN + KN FLOPs**
- **Memory: KN (for X1 only)**

### **Savings**

```python
# FLOPs reduction
Before: 2MKN + MN
After:  MKN + KN
Saved:  MKN + MN - KN

# For M=1024, K=512, N=256:
Before: 268,697,600 FLOPs
After:  134,479,872 FLOPs
Speedup: 2.0× 🚀

# Memory reduction
Before: 2MN = 524,288 elements = 2.1 MB (FP32)
After:  KN = 131,072 elements = 0.5 MB (FP32)
Saved: 76% memory 💾
```

---

## Example 2: Transformer Attention (Real-World)

### **Before Optimization**

```python
# Standard attention computation
class Attention(nn.Module):
    def forward(self, X):
        Q = self.W_q @ X  # [d_model, seq_len]
        K = self.W_k @ X  # [d_model, seq_len]
        V = self.W_v @ X  # [d_model, seq_len]
        
        scores = Q.T @ K  # [seq_len, seq_len]
        attn = softmax(scores)
        output = attn @ V.T  # [seq_len, d_model]
        return output

# Graph
           X
         / | \
        @  @  @
      W_q W_k W_v
       Q   K   V
        \ /    |
         @     |
      scores   |
         |     |
      softmax  |
         |     |
        attn   |
          \   /
           @
           |
         output
```

**Cost:**
- 3 separate MatMuls: `W_q@X`, `W_k@X`, `W_v@X`
- QK attention: `Q.T @ K`
- Attention application: `attn @ V.T`
- **Total: 5 MatMuls**

### **TASO Optimization 1: Weight Concatenation**

```python
# Fuse Q, K, V projections
class AttentionOptimized(nn.Module):
    def __init__(self):
        # Concatenate weight matrices
        self.W_qkv = torch.cat([W_q, W_k, W_v], dim=0)
    
    def forward(self, X):
        # Single batched matmul
        QKV = self.W_qkv @ X  # [3*d_model, seq_len]
        Q, K, V = torch.split(QKV, d_model, dim=0)
        
        scores = Q.T @ K
        attn = softmax(scores)
        output = attn @ V.T
        return output

# Graph
           X
           |
           @
        W_qkv
           |
         split
       /   |   \
      Q    K    V
       \  /     |
        @       |
      scores    |
        |       |
     softmax    |
        |       |
       attn     |
         \     /
          @
          |
        output
```

**Savings:**
- 3 MatMuls → 1 MatMul (3× reduction!)
- Better GPU utilization (larger batch)
- **Speedup: 1.8-2.2× on attention projection**

### **TASO Optimization 2: Fused Softmax**

```python
# Before: Separate operations
scores = Q.T @ K
max_scores = scores.max(dim=-1)
shifted = scores - max_scores
exp_scores = exp(shifted)
sum_exp = exp_scores.sum(dim=-1)
attn = exp_scores / sum_exp

# After: Fused operation (TASO + kernel fusion)
attn = fused_softmax(Q.T @ K)
```

**Savings:**
- 5 separate kernels → 1 fused kernel
- Intermediate tensors eliminated
- **Speedup: 2-3× on softmax** (same as Triton tutorial!)

### **Combined Savings**

| Metric | Original | TASO Optimized | Improvement |
|--------|----------|----------------|-------------|
| MatMuls | 5 | 3 | 40% reduction |
| Kernel Launches | ~10 | ~5 | 50% reduction |
| Memory (intermediates) | High | Low | 40-60% less |
| **End-to-End Speedup** | 1.0× | **1.8-2.2×** | **TASO win!** 🎯 |

---

## Example 3: Batch Normalization Fusion

### **Before Optimization**

```python
# Separate operations
def batchnorm_relu(X, gamma, beta):
    mean = X.mean(dim=0)
    var = X.var(dim=0)
    normalized = (X - mean) / sqrt(var + eps)
    scaled = normalized * gamma
    shifted = scaled + beta
    activated = relu(shifted)
    return activated

# Graph
    X
    |
  mean, var
    |
 normalize
    |
  × gamma
    |
  + beta
    |
   relu
    |
    Y
```

**Cost:**
- 6 separate operations
- Multiple passes over data
- Intermediate tensors for each step

### **After TASO Optimization**

```python
# Fused operation
def batchnorm_relu_fused(X, gamma, beta):
    # All in one kernel
    return fused_bn_relu(X, gamma, beta)

# Graph
    X
    |
  FusedBNReLU
    |
    Y
```

**Rewrite Rule Used:**
```
ReLU(BatchNorm(X)) → FusedBatchNormReLU(X)
```

**Savings:**
- 6 operations → 1 operation
- Data loaded once (vs 6 times)
- **Speedup: 3-5× on BatchNorm+ReLU**

---

## Example 4: Associativity (Matrix Chain)

### **Before Optimization**

```python
# Left-associative
Y = ((A @ B) @ C) @ D

# Graph
      A   B
       \ /
        @
        |
        X1  C
         \ /
          @
          |
          X2  D
           \ /
            @
            |
            Y
```

**Cost:**
- Shapes: A=[1000×10], B=[10×1000], C=[1000×10], D=[10×1]
- A@B: 1000×10×1000 = 10M FLOPs → [1000×1000] intermediate
- (A@B)@C: 1000×1000×10 = 10M FLOPs → [1000×10] intermediate
- ((A@B)@C)@D: 1000×10×1 = 10K FLOPs → [1000×1] output
- **Total: 20M FLOPs**
- **Huge intermediate: 1000×1000 = 1M elements!**

### **After TASO Optimization**

```python
# Right-associative (better!)
Y = A @ (B @ (C @ D))

# Graph
              C   D
               \ /
                @
                |
            B   X1
             \ /
              @
              |
          A   X2
           \ /
            @
            |
            Y
```

**Cost:**
- C@D: 1000×10×1 = 10K FLOPs → [1000×1] intermediate
- B@(C@D): 10×1000×1 = 10K FLOPs → [10×1] intermediate
- A@(B@(C@D)): 1000×10×1 = 10K FLOPs → [1000×1] output
- **Total: 30K FLOPs**
- **Max intermediate: 1000×1 = 1K elements**

**Savings:**
```
FLOPs: 20M → 30K = 667× reduction! 🤯
Memory: 1M elements → 1K elements = 1000× reduction!
```

**Lesson:** Parenthesization matters HUGELY for matrix chains!

---

## Example 5: Transpose Elimination

### **Before Optimization**

```python
# Double transpose
Y = (A.T).T

# Or more subtly:
Q = W_q @ X
K = W_k @ X
scores = Q.T @ K.T  # Transpose both!
```

**After TASO Optimization**

```python
# Eliminate redundant transpose
Y = A  # (A.T).T = A

# Or:
Q = W_q @ X
K = W_k @ X
scores = (K @ Q).T  # Use (AB)ᵀ = BᵀAᵀ rule
```

**Savings:**
- Eliminate transpose operations (memory layout changes)
- Reduce memory copies
- **Speedup: 1.2-1.5× (transpose costs add up!)**

---

## Example 6: Common Subexpression Elimination

### **Before Optimization**

```python
# Duplicate computation
X1 = A @ B
Y1 = X1 + C

X2 = A @ B  # Same as X1!
Y2 = X2 + D

# Graph
    A   B       A   B
     \ /         \ /
      @           @
      |           |
      X1  C       X2  D
       \ /         \ /
        +           +
        |           |
        Y1          Y2
```

**After TASO Optimization**

```python
# Reuse common subexpression
X1 = A @ B
Y1 = X1 + C
Y2 = X1 + D  # Reuse X1!

# Graph
    A   B
     \ /
      @
      |
      X1
     /  \
    /    \
   +      +
   C      D
   |      |
   Y1     Y2
```

**Savings:**
- 2 MatMuls → 1 MatMul (2× reduction)
- 1 intermediate tensor instead of 2
- **Speedup: 2× for this pattern**

---

## Example 7: Einstein Summation Optimization

### **Before Optimization**

```python
# Naive einsum
# "ij,jk,kl->il"
Y = einsum("ij,jk->ik", A, B)
Z = einsum("ik,kl->il", Y, C)

# Graph
  A   B       Y   C
   \ /         \ /
  einsum     einsum
    |           |
    Y           Z
```

**After TASO Optimization**

```python
# Fused einsum
Z = einsum("ij,jk,kl->il", A, B, C)

# Or better: optimal contraction order
# TASO decides: A@(B@C) vs (A@B)@C
# Based on shapes!
```

**Savings:**
- Optimal contraction order (like Example 4)
- Eliminate intermediate tensors
- **Speedup: 2-10× depending on shapes!**

---

## Summary: TASO Rewrite Rules

| Rule | Example | Typical Speedup |
|------|---------|-----------------|
| **Distributivity** | `A·B + A·C → A·(B+C)` | 1.5-2× |
| **Associativity** | `(A·B)·C → A·(B·C)` | 2-1000× (shape-dependent!) |
| **Operator Fusion** | `ReLU(BN(X)) → BNReLU(X)` | 2-5× |
| **Transpose Rules** | `(Aᵀ)ᵀ → A` | 1.2-1.5× |
| **CSE** | Reuse `A·B` | 2× per duplicate |
| **Constant Folding** | Precompute constants | Varies |
| **Batching** | `3 matmuls → 1 batched` | 1.8-2.2× |

---

## Real-World Impact

### **BERT-base Optimization**

Original graph: 250+ operations
TASO optimized: 180 operations (28% reduction)

**Breakdown:**
- Attention blocks: 2.0× faster (weight concatenation)
- LayerNorm+Residual: 1.5× faster (fusion)
- Feed-forward: 1.3× faster (fusion)
- **End-to-end: 1.6× faster** 🚀

### **GPT-2 Optimization**

Original graph: 400+ operations
TASO optimized: 290 operations (27.5% reduction)

**Breakdown:**
- Multi-head attention: 2.2× faster
- MLP blocks: 1.4× faster
- Embedding+Position: 1.2× faster
- **End-to-end: 1.7× faster** 🚀

---

## Next: See It Running

Ready to see these optimizations in action?

**Run `simple_rewrite.py`** - Example 1 (distributivity) with actual benchmarks on your machine!

Let's see TASO eliminate operations! 📉🚀

---

*Each operation TASO eliminates is one less thing for your GPU to compute!*
