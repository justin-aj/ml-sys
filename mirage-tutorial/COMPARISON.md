# Mirage vs Other Optimizers: Complete Comparison

This document compares Mirage's equality saturation approach with all other optimization tools in this repository.

---

## 🎯 Quick Reference Table

| Tool | Level | Approach | Search Strategy | Optimality | Speed | Production |
|------|-------|----------|-----------------|------------|-------|------------|
| **Mirage** | Graph | Equality saturation | Exhaustive | Global | Slow (mins) | ❌ Research |
| **TASO** | Graph | Rule-based | Greedy/Beam | Local | Fast (secs) | ✅ Yes (Microsoft) |
| **Ansor** | Schedule | ML-guided | Sampling | Near-optimal | Medium (hours) | ⚠️ Via TVM |
| **Triton** | Kernel | Manual | User-defined | User-skill | Instant | ✅ Yes (OpenAI) |
| **Mega-Kernels** | Kernel | Manual CUDA | User-defined | User-skill | Instant | ✅ Educational |

---

## Mirage vs TASO

### **Core Difference: Exhaustive vs Greedy**

**TASO:**
```
Input: A @ B + A @ C

Step 1: See distributive rule applies
Step 2: Apply: A @ B + A @ C → A @ (B + C)
Step 3: Done! Move to next opportunity

Result: Good, but what if there was a better path?
```

**Mirage:**
```
Input: A @ B + A @ C

Step 1: Build e-graph with ALL equivalents:
  - A @ B + A @ C
  - A @ (B + C)        [distributive]
  - (B + C) @ A        [commutative]
  - B @ A + C @ A      [distributive + commutative]
  - ... hundreds more

Step 2: Pick cheapest from ALL options

Result: Globally optimal (within rewrite rules)
```

### **Performance Comparison**

| Metric | TASO | Mirage |
|--------|------|--------|
| **Time** | Seconds | Minutes |
| **Search Space** | O(n²) programs | Exponential (compactly) |
| **Optimality** | Local minimum | Global optimum |
| **Novel Patterns** | ❌ Only predefined | ✅ Can discover |

**Example: BERT Optimization**
```
TASO:   1.6x speedup (applies known rules)
Mirage: 1.9x speedup (finds non-obvious combinations)

Cost: TASO 5 seconds, Mirage 2 minutes
```

### **When to Use What**

**Use TASO when:**
- ✅ Need fast iteration (development)
- ✅ Production deployment (proven, mature)
- ✅ "Good enough" is acceptable

**Use Mirage when:**
- ✅ Research/exploration ("what's possible?")
- ✅ Critical optimization (want absolute best)
- ✅ Discovering new patterns to add to TASO

---

## Mirage vs Ansor

### **Core Difference: Graph vs Schedule**

**Ansor:**
```
Takes: Computation graph (already optimized)
Optimizes: Loop schedules (tiling, parallelization)

Example:
  Input: matmul(A, B)
  Output: Optimal loop tiling + thread mapping for GPU

Doesn't change: The computation itself
```

**Mirage:**
```
Takes: Computation graph
Optimizes: The computation structure itself

Example:
  Input: matmul(A, B) + matmul(A, C)
  Output: matmul(A, add(B, C))  # Different computation!

Changes: Number and order of operations
```

### **Complementary, Not Competing**

```
Ideal Pipeline:

Input Model
    │
    ▼
┌─────────────────────────────┐
│ Mirage: Graph Optimization  │  ← Change computation structure
│ A@B + A@C → A@(B+C)         │     Reduce operations
└───────────┬─────────────────┘
            │
            ▼
┌─────────────────────────────┐
│ Ansor: Schedule Optimization│  ← Optimize remaining operations
│ Find optimal loop tiling    │     Auto-tune for hardware
└───────────┬─────────────────┘
            │
            ▼
      Optimal Binary
```

**Combined Speedup:**
```
Mirage alone:  1.8x
Ansor alone:   1.4x
Combined:      2.5x (multiplicative!)
```

### **Search Strategy Comparison**

| Aspect | Ansor | Mirage |
|--------|-------|--------|
| **Space** | Loop schedules (10^10+ options) | Equivalent graphs (exponential) |
| **Method** | ML-guided sampling (XGBoost) | Exhaustive enumeration (e-graph) |
| **Correctness** | Heuristic (might miss optimal) | Proven (finds optimal) |
| **Speed** | Hours of tuning | Minutes of search |

---

## Mirage vs Triton

### **Core Difference: Automatic vs Manual**

**Triton:**
```python
# You write the fused kernel manually
@triton.jit
def fused_softmax(x_ptr, output_ptr, ...):
    # Load x
    x = tl.load(x_ptr + offsets)
    
    # Compute in registers (you design this!)
    max_val = tl.max(x)
    shifted = x - max_val
    exp_x = tl.exp(shifted)
    sum_exp = tl.sum(exp_x)
    softmax = exp_x / sum_exp
    
    # Store output
    tl.store(output_ptr + offsets, softmax)

Result: 3x faster (if you write it well!)
```

**Mirage:**
```python
# Mirage discovers the fusion automatically
softmax(x)

# Mirage explores:
# - Decomposed form: exp(x) / sum(exp(x))
# - Stable form: exp(x - max(x)) / sum(exp(x - max(x)))
# - Fused form: fused_softmax(x)
# Picks: fused_softmax (optimal)

Result: Tells you WHAT to fuse (then use Triton to implement!)
```

### **Complementary Workflow**

```
Step 1: Use Mirage
  → Discovers optimal computation graph
  → Identifies fusion opportunities
  → Proves correctness of transformations

Step 2: Use Triton
  → Implement the fused kernels Mirage suggested
  → Write high-performance Python code
  → Deploy to production

Example:
  Mirage: "Fuse softmax + matmul for 2x speedup"
  You: Write Triton kernel implementing fused_softmax_matmul
  Result: Best of both worlds!
```

### **Design Philosophy**

| Aspect | Triton | Mirage |
|--------|--------|--------|
| **Who optimizes** | You (programmer) | Computer (automatic) |
| **What you control** | Everything | Nothing (provides answer) |
| **Learning curve** | Moderate (GPU concepts) | Steep (equality saturation) |
| **Iteration speed** | Instant (write → run) | Slow (minutes of search) |
| **Creativity** | Unlimited | Limited to rewrite rules |

---

## Mirage vs Mega-Kernels

### **Core Difference: Discovery vs Implementation**

**Mega-Kernels (CUDA):**
```cuda
// You manually fuse GELU + scale
__global__ void gelu_scale_fused(float* out, float* in, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        
        // GELU in registers
        float gelu = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x*x*x)));
        
        // Scale in registers
        out[idx] = gelu * scale;  // Fused!
    }
}

Result: 1.9x faster (hand-optimized CUDA)
```

**Mirage:**
```python
# Input
y = scale * gelu(x)

# Mirage explores:
# - Sequential: gelu(x) then scale
# - Fused: gelu_scale(x)
# - Reordered: scale * gelu vs gelu * scale (equivalent)
# Picks: fused_gelu_scale (optimal)

Result: Tells you TO fuse, not HOW to implement
```

### **Relationship**

```
Mirage: Strategy designer
  "You should fuse these 3 operations"
  
Mega-Kernels/Triton: Implementation
  "Here's how to fuse them in CUDA/Triton"
  
Combined:
  Mirage decides WHAT
  You implement HOW
```

---

## The Complete Optimization Stack (Redux)

### **All Tools Together**

```
┌──────────────────────────────────────────────────────────┐
│ 1. GRAPH STRUCTURE (Mirage vs TASO)                     │
│                                                          │
│  Mirage: Exhaustive search → Global optimum             │
│    Pros: Finds absolute best, discovers patterns        │
│    Cons: Slow (minutes), research-only                  │
│                                                          │
│  TASO: Greedy search → Local optimum                    │
│    Pros: Fast (seconds), production-ready               │
│    Cons: Might miss global optimum                      │
│                                                          │
│  Speedup: 1.5-2x (Mirage slight edge)                   │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│ 2. KERNEL FUSION (Triton vs Mega-Kernels)               │
│                                                          │
│  Triton: Manual fusion in Python                        │
│    Pros: Production-ready, good performance             │
│    Cons: You design the fusion                          │
│                                                          │
│  Mega-Kernels: Manual fusion in CUDA                    │
│    Pros: Maximum control, educational                   │
│    Cons: Harder to write, CUDA knowledge needed         │
│                                                          │
│  Speedup: 1.3-1.5x (both similar)                       │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│ 3. SCHEDULE TUNING (Ansor)                              │
│                                                          │
│  Ansor: ML-guided auto-tuning                           │
│    Pros: Automated, learns across devices               │
│    Cons: Requires TVM, hours of tuning                  │
│                                                          │
│  Speedup: 1.2-1.5x                                      │
└──────────────────────────────────────────────────────────┘
                     │
                     ▼
              FINAL BINARY
         (2-5x faster combined!)
```

---

## Production Decision Matrix

### **Which Tool for Which Scenario**

| Your Situation | Recommendation | Reasoning |
|----------------|----------------|-----------|
| **Research: "What's possible?"** | Mirage | Exhaustive search, novel patterns |
| **Production: Fast transformer** | TASO + Triton | Fast, proven, production-ready |
| **Custom model optimization** | Mirage → analyze → Triton | Discover, then implement |
| **Learning GPU optimization** | Mega-Kernels → Triton | Understand concepts, then production |
| **Cross-device deployment** | TASO + Ansor | Graph + schedule optimization |
| **Rapid iteration** | TASO | Seconds vs minutes |
| **Absolute best performance** | Mirage + Triton + Ansor | All three levels! |

---

## Real-World Example: Optimizing BERT

### **Scenario:** Optimize BERT-base for production inference

#### **Approach 1: TASO Only** ✅
```
Time: 10 seconds
Speedup: 1.6x
Deployment: Immediate
Cost: Free (open source)

Good for: Quick wins, production timelines
```

#### **Approach 2: Mirage Discovery** 🔬
```
Time: 5 minutes search
Speedup: 1.9x
Deployment: Extract patterns → add to TASO
Cost: Research time

Good for: Finding novel patterns, research
```

#### **Approach 3: Triton Kernels** ⚡
```
Time: 2 days writing kernels
Speedup: 1.7x (on top of TASO)
Deployment: Package Triton kernels
Cost: Engineering time

Good for: Production systems, critical paths
```

#### **Approach 4: Everything** 🚀
```
Time: Mirage (5 min) + Triton (2 days) + Ansor (overnight)
Speedup: 2.8x combined
Deployment: Complex pipeline
Cost: High engineering investment

Good for: Critical production systems, maximum performance
```

**Practical Choice:** TASO + Triton (best speed/effort tradeoff)

---

## Conceptual Advantages of Each

### **Mirage's Unique Value**

✅ **Provably optimal** (within rewrite rules)
✅ **Discovers patterns** humans miss
✅ **Guarantees correctness** (mathematical equivalence)
✅ **Research tool** for finding new optimizations

### **TASO's Unique Value**

✅ **Production-ready** (used by Microsoft)
✅ **Fast iteration** (seconds)
✅ **Good enough** (local optimal often sufficient)
✅ **Mature ecosystem**

### **Triton's Unique Value**

✅ **Easy to write** (Python, not CUDA)
✅ **Production performance** (90-100% of CUDA)
✅ **Active community** (OpenAI, Meta, HuggingFace)
✅ **pip install** (actually works!)

### **Ansor's Unique Value**

✅ **Transfer learning** (GPU A → GPU B)
✅ **Automated tuning** (no manual work)
✅ **Schedule space exploration** (exhaustive at schedule level)

---

## Future: Convergence?

### **The Ideal Future Tool Would:**

✅ **Mirage's exhaustive search** (find optimal)
✅ **TASO's speed** (seconds, not minutes)
✅ **Triton's ease** (Python-based)
✅ **Ansor's learning** (adapt across devices)

**Research Directions:**
- Faster equality saturation (making Mirage production-ready)
- Learned cost models (Ansor-style ML for Mirage)
- Integrated pipelines (Mirage → Triton → Ansor automatic)

**Current Reality:**
- Use Mirage for research/discovery
- Use TASO + Triton for production
- Use Ansor when you have TVM infrastructure

---

## Summary Table: The Complete Picture

| Dimension | Mirage | TASO | Triton | Mega-Kernels | Ansor |
|-----------|--------|------|--------|--------------|-------|
| **Speed** | ⭐ (slow) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ (hours) |
| **Optimality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ (user) | ⭐⭐⭐⭐ (user) | ⭐⭐⭐⭐ |
| **Automation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ (manual) | ⭐ (manual) | ⭐⭐⭐⭐⭐ |
| **Discovery** | ⭐⭐⭐⭐⭐ | ⭐ (rules) | ⭐⭐ (user) | ⭐⭐ (user) | ⭐⭐ (schedules) |
| **Production** | ⭐ (research) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ (via TVM) |
| **Learning Curve** | ⭐ (hard) | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ (CUDA) | ⭐⭐ (TVM) |

---

**The Bottom Line:**

- **Mirage** = Best for research, discovering what's possible
- **TASO** = Best for production graph optimization
- **Triton** = Best for writing fast kernels in practice
- **Mega-Kernels** = Best for learning GPU fundamentals
- **Ansor** = Best when you have TVM infrastructure

**Stack them for maximum performance: 2-5x speedup!**

---

*Each tool has its place in the optimization toolkit. The best approach depends on your goals, timeline, and constraints.*
