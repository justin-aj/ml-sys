# TASO Concept: Graph-Level Optimization

## 🎯 The Big Idea

**TASO optimizes computation graphs using algebraic rewrites.**

Instead of making individual operations faster (like Triton), TASO **eliminates or combines operations** using mathematical identities.

---

## 📝 Concrete Example: The Distributive Property

### **Starting Graph**

```python
Y = (A · B) + (A · C)
```

**Computation Graph:**
```
    A
   / \
  ·   ·
 B     C
  \   /
   +
   |
   Y
```

**Cost:**
- 2 matrix multiplications (expensive!)
- 1 addition
- 2 intermediate tensors stored in memory

---

### **TASO Rewrites It**

Using the distributive property: `A·B + A·C = A·(B+C)`

```python
Y = A · (B + C)
```

**Optimized Graph:**
```
    B   C
     \ /
      +
      |
      ·
      A
      |
      Y
```

**Cost:**
- 1 matrix multiplication (50% reduction!)
- 1 addition
- 1 intermediate tensor

**Savings:**
- **50% fewer FLOPs** (1 matmul vs 2 matmuls)
- **50% less memory** (1 intermediate vs 2)
- **Fewer kernel launches** (2 GPU kernels vs 3)

---

## 🔍 How TASO Finds This

### **Step 1: Input Graph**
```python
# User writes code
X1 = torch.matmul(A, B)
X2 = torch.matmul(A, C)
Y = X1 + X2
```

### **Step 2: Apply Rewrite Rules**

TASO has a library of algebraic identities:

| Rule | Transformation |
|------|----------------|
| **Distributivity** | `A·B + A·C → A·(B+C)` |
| **Associativity** | `(A·B)·C → A·(B·C)` |
| **Commutativity** | `A+B → B+A` |
| **Fusion** | `ReLU(A+B) → ReLU_Add(A,B)` |

### **Step 3: Graph Search**

```
Original Graph
      │
      ▼
  Apply all rules
      │
      ├──► Candidate 1: A·(B+C)     [Cost: low]
      ├──► Candidate 2: (A·B)+(A·C)  [Cost: high]
      └──► Candidate 3: ...
      │
      ▼
  Select lowest cost
      │
      ▼
Optimized Graph: A·(B+C)
```

### **Step 4: Cost Estimation**

TASO estimates cost using:
- **FLOPs:** How many operations?
- **Memory:** How many intermediate tensors?
- **Hardware model:** Which GPU? Memory bandwidth?

**Example Cost Model:**
```python
def cost(graph):
    flops = sum(op.flops for op in graph.operations)
    memory = sum(tensor.size for tensor in graph.intermediates)
    kernel_launches = len(graph.operations)
    
    return α*flops + β*memory + γ*kernel_launches
```

---

## 🧮 Real Numbers: Transformer Attention

### **Original Attention (Simplified)**

```python
# Attention mechanism
Q = Linear1(X)  # matmul
K = Linear2(X)  # matmul
V = Linear3(X)  # matmul
scores = Q @ K.T  # matmul
attn = softmax(scores)
output = attn @ V  # matmul
```

**Cost:** 5 matrix multiplications

### **TASO Optimization**

TASO notices that `Q`, `K`, `V` all multiply the same input `X`:

```python
# Before: 3 separate matmuls
Q = W_Q @ X
K = W_K @ X
V = W_V @ X

# After: 1 batched matmul (TASO rewrite)
QKV = [W_Q; W_K; W_V] @ X  # Concatenated weight matrix
Q, K, V = split(QKV)
```

**Savings:**
- 3 matmuls → 1 matmul (3× reduction!)
- Better GPU utilization (larger batched operation)
- Fewer kernel launches

**Real Speedup:** 1.8-2.2× faster on NVIDIA GPUs (measured)

---

## 📊 TASO vs Other Optimizers

### **PyTorch JIT (TorchScript)**

```python
# PyTorch does some fusion
@torch.jit.script
def forward(A, B, C):
    return A @ B + A @ C

# Result: Limited fusion (add+matmul maybe)
# Speedup: 1.1-1.3×
```

**Limitation:** Heuristic-based, doesn't explore algebraic rewrites

### **ONNX Runtime**

```python
# ONNX Runtime has fusion patterns
# E.g., Gemm+Add fusion, Conv+BatchNorm fusion

# Speedup: 1.2-1.5×
```

**Limitation:** Fixed fusion patterns, not exhaustive search

### **TASO**

```python
# TASO exhaustively searches algebraic rewrites
optimized = taso.optimize(graph, alpha=1.0, beta=0.5)

# Speedup: 1.5-2.5× (finds non-obvious optimizations!)
```

**Advantage:** Mathematical correctness + exhaustive search = finds optimizations others miss

---

## 🔬 TASO's Rewrite Rules (Examples)

### **1. Linear Algebra Identities**

```python
# Distributivity
A·B + A·C → A·(B+C)

# Associativity
(A·B)·C → A·(B·C)

# Transpose
(A·B)ᵀ → Bᵀ·Aᵀ
```

### **2. Operator Fusion**

```python
# Add + ReLU
ReLU(A + B) → AddReLU(A, B)

# BatchNorm + ReLU
ReLU(BatchNorm(X)) → BatchNormReLU(X)

# Softmax decomposition
Softmax(X) → Exp(X - Max(X)) / Sum(Exp(X - Max(X)))
```

### **3. Constant Folding**

```python
# Compile-time evaluation
Y = X · W  where W is constant
→ Precompute parts involving W
```

### **4. Redundancy Elimination**

```python
# Common subexpression elimination
X1 = A · B
X2 = A · B  # Duplicate!
→ X1 = A · B; X2 = X1  # Reuse
```

---

## 🎯 When Does TASO Win Big?

### **Best For:**

1. **Transformer Models** (attention has lots of matmul patterns)
   - BERT: 1.5-1.8× speedup
   - GPT-2: 1.6-2.0× speedup
   - Attention blocks: 2-3× speedup

2. **Custom Architectures** (non-standard patterns PyTorch doesn't optimize)
   - Research models with novel operators
   - Domain-specific neural networks

3. **Memory-Constrained Deployment**
   - Edge devices (reduce memory footprint 30-50%)
   - Multi-model serving (fit more models in memory)

### **Less Effective For:**

1. **Simple Sequential Models** (ResNet, VGG)
   - Limited algebraic rewrite opportunities
   - Speedup: 1.1-1.3× (modest)

2. **Single Large Operations** (one giant matmul)
   - No graph-level optimization possible
   - Use kernel-level optimization (Triton) instead

---

## 🔄 The Optimization Stack

```
┌──────────────────────────────────────────┐
│  MODEL LEVEL (Architecture)              │  ← Design choices
│  (e.g., use multi-query attention)       │
└────────────────┬─────────────────────────┘
                 │
┌────────────────▼─────────────────────────┐
│  GRAPH LEVEL (TASO)                      │  ← Algebraic rewrites
│  A·B + A·C → A·(B+C)                     │     1.5-2× speedup
└────────────────┬─────────────────────────┘
                 │
┌────────────────▼─────────────────────────┐
│  KERNEL LEVEL (Triton)                   │  ← Fusion + memory opt
│  Fuse softmax operations                 │     1.3-1.5× speedup
└────────────────┬─────────────────────────┘
                 │
┌────────────────▼─────────────────────────┐
│  SCHEDULE LEVEL (Ansor)                  │  ← Auto-tune loops
│  Find optimal tile sizes                 │     1.2-1.5× speedup
└────────────────┬─────────────────────────┘
                 │
                 ▼
           FINAL PERFORMANCE
         (3-5× faster combined!)
```

**Key Insight:** Each level optimizes different aspects. Stack them for maximum performance!

---

## 💡 TASO in Production

### **Microsoft**
- Uses TASO for model serving in Azure ML
- Optimizes customer models automatically
- Report: 1.5-2× average speedup on transformers

### **OctoML**
- TASO + TVM for cross-device optimization
- Optimizes same model for 100+ device types
- Reduces deployment time from weeks to hours

### **Facebook/Meta (Research)**
- Explored TASO for PyTorch graph optimization
- Found 1.3-2× speedups on production models
- Some ideas integrated into TorchScript

---

## 🎓 Key Takeaways

1. **Graph optimization** is orthogonal to kernel optimization
   - TASO: Reduce number of operations
   - Triton: Make each operation faster
   - **Use both!**

2. **Algebraic rewrites** can find non-obvious optimizations
   - `A·B + A·C → A·(B+C)` seems simple
   - But TASO finds **hundreds** of such patterns in real models

3. **Mathematical correctness** ensures safety
   - All TASO rewrites are mathematically equivalent
   - No approximations (unlike some quantization techniques)

4. **Hardware-aware** optimization matters
   - Same algebraic rewrite may be beneficial on GPU A but not GPU B
   - TASO uses cost models to choose platform-specific optimizations

---

## 📚 Next: See It In Action

Ready to see concrete examples?

1. **`EXAMPLES.md`** - Walk through detailed rewrite examples
2. **`simple_rewrite.py`** - Run the A·B + A·C example
3. **`transformer_attention.py`** - Optimize real attention block

Let's see TASO eliminate operations! 🚀

---

*"The fastest operation is the one you never execute."* — TASO Philosophy
