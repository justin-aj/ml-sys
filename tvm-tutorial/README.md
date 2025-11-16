# TVM Overview - Deep Learning Compiler

**Note:** TVM is difficult to install and set up. This directory contains **conceptual documentation** about how TVM works, without requiring installation.

For practical GPU optimization tutorials, see: **[../mega-kernels/](../mega-kernels/)** ✅

For easier alternatives to TVM, see: **[TVM_ALTERNATIVES.md](TVM_ALTERNATIVES.md)** 🚀

---

## 🎯 What is TVM?

**TVM (Tensor Virtual Machine)** is an open-source deep learning compiler that optimizes ML models for any hardware.

### Core Idea

```
PyTorch/TensorFlow Model
         ↓
    TVM Compiler
         ↓
Optimized Code for: NVIDIA GPU | AMD GPU | ARM | x86 | TPU | Custom Hardware
```

**Write once, deploy anywhere** - with automatic optimization for each platform.

---

## 📚 Key Concepts (No Installation Required)

### 1. Tensor Expressions - Define WHAT to Compute

Instead of writing imperative code, you declare the computation:

```python
# Conceptual example (doesn't require TVM installed)
# Matrix multiplication definition
C[i, j] = sum(A[i, k] * B[k, j] for k in range(K))
```

TVM's Tensor Expression language lets you write this declaratively.

### 2. Schedules - Define HOW to Compute

The **schedule** specifies optimization strategy without changing the algorithm:

```python
# Conceptual example
# Basic schedule (slow):
for i in range(M):
    for j in range(N):
        C[i,j] = ...

# Optimized schedule (fast):
for i_outer in range(M // 32):      # Tiling
    for j_outer in range(N // 32):
        for i_inner in range(32):    # Cache-friendly blocks
            for j_inner in range(32):
                C[i_outer*32 + i_inner, j_outer*32 + j_inner] = ...
```

**Same computation, 5x faster** through better memory access patterns.

### 3. Auto-Tuning - Find Optimal Schedule Automatically

TVM uses machine learning to search for the best schedule:

1. Generate candidate schedules (different tile sizes, memory strategies)
2. Compile and benchmark each on your actual hardware
3. Learn which schedules perform best
4. Iterate to find optimal configuration

**Result:** Often beats hand-written CUDA kernels!

---

## 🧠 Detailed Concepts

### 1. Tensor Expressions - WHAT to Compute

Define the computation algorithm without specifying implementation:

```python
from tvm import te

# Element-wise addition
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute((n,), lambda i: A[i] + B[i], name="C")

# Matrix multiplication
k = te.reduce_axis((0, K), name="k")
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k))
```

**Key idea:** Declarative - you describe WHAT, not HOW.

### 2. Schedules - HOW to Compute

Specify optimization strategy without changing the algorithm:

```python
# Basic schedule (slow)
s = te.create_schedule(C.op)

# Optimized schedule (fast)
x, y = C.op.axis
xo, xi = s[C].split(x, factor=32)  # Tiling
s[C].parallel(xo)                   # Multi-threading
s[C].vectorize(xi)                  # SIMD instructions
```

**Key idea:** Same computation, different performance.

### 3. Auto-Tuning - Find Best Schedule

Let TVM search for optimal schedules automatically:

```python
from tvm import auto_scheduler

# Define task
task = auto_scheduler.SearchTask(func=matmul, args=(M, N, K), target="cuda")

# Search for best schedule
task.tune(auto_scheduler.TuningOptions(num_measure_trials=1000))

# Apply best schedule
sch, args = task.apply_best("tune.log")
```

**Key idea:** ML-guided search finds schedules you might not think of.

---

## 🎓 Learning Path

### Beginner (1-2 hours)

1. ✅ Read this README and LEARNING_GUIDE.md
2. ✅ Understand Tensor Expressions and Schedules concepts
3. ✅ Review schedule primitives table
4. ✅ Compare TVM approach with Triton/PyTorch alternatives

**Goal:** Understand separation of algorithm and optimization.

### Intermediate (2-4 hours)

1. ✅ Read LEARNING_GUIDE.md (GPU Optimization section)
2. ✅ Study GPU schedule primitives (split, bind, cache)
3. ✅ Explore REAL_WORLD_USES.md for production examples
4. ✅ Compare TVM concepts with mega-kernels tutorial

**Goal:** Master GPU schedule primitives conceptually.

### Advanced (4-8 hours)

1. ✅ Read REAL_WORLD_USES.md for production case studies
2. ✅ Study auto-tuning concepts in LEARNING_GUIDE.md
3. ✅ Explore TVM_ALTERNATIVES.md for practical implementations
4. ✅ Implement similar concepts using Triton or torch.compile()

**Goal:** Understand production ML compilation strategies.

---

## 📊 Expected Performance

### Expected CPU Performance

Matrix multiplication (512x512):

| Implementation | Time | Speedup |
|----------------|------|---------|
| PyTorch | 2.45 ms | 1.0x (baseline) |
| TVM (basic) | 4.20 ms | 0.6x (slower) |
| TVM (optimized) | 0.98 ms | **2.5x faster** |

**Optimizations:** Tiling (32x32) + Vectorization + Parallelization

### Expected GPU Performance

Matrix multiplication (2048x2048) on V100:

| Implementation | Time | GFLOPS | Speedup |
|----------------|------|--------|---------|
| PyTorch (cuBLAS) | 1.85 ms | 9305 | 1.0x (baseline) |
| TVM (basic) | 8.50 ms | 2024 | 0.2x (much slower) |
| TVM (optimized) | 2.10 ms | 8191 | **0.88x** (competitive!) |

**Optimizations:** Shared memory + Register blocking + Thread cooperation

**Note:** cuBLAS is extremely well-tuned. TVM shines on custom/fused operations.

### Auto-Tuning Performance

With 1000 trials of auto-tuning, expect:
- **1.5-2x** improvement over manual schedule
- **Competitive with cuBLAS** for standard ops
- **Better than cuBLAS** for custom fused operations

---

## 🛠️ Schedule Primitives (Conceptual)

| Primitive | Purpose | Example |
|-----------|---------|---------|
| `split(axis, factor)` | Divide loop into outer/inner | Cache blocking |
| `tile(x, y, xf, yf)` | 2D tiling | Matrix operations |
| `reorder(*axes)` | Change loop order | Memory access patterns |
| `bind(axis, thread)` | Map to GPU threads | GPU parallelization |
| `vectorize(axis)` | Use SIMD | CPU vector instructions |
| `parallel(axis)` | Multi-threading | CPU multi-core |
| `unroll(axis)` | Loop unrolling | Reduce overhead |
| `cache_read/write` | Memory hierarchy | GPU shared memory |

**See LEARNING_GUIDE.md for detailed explanations and examples.**

---

## 🌍 Real-World Applications

**See [REAL_WORLD_USES.md](REAL_WORLD_USES.md) for detailed examples.**

TVM is used in production by:

### Companies
- **AWS** - Inferentia chips, SageMaker Neo
- **Meta** - Transformer inference (billions of inferences/day)
- **AMD** - ROCm GPU stack
- **OctoML** - ML deployment platform
- **Arm** - Mobile and edge AI

### Use Cases
- **Computer Vision** - Deploy YOLOv5 to Raspberry Pi
- **NLP** - Optimize BERT for low-latency inference
- **Recommendations** - 3x throughput for DLRM models
- **Speech** - Real-time ASR on mobile devices

**See REAL_WORLD_USES.md for detailed case studies.**

---

## ⚠️ Why Not Use TVM?

### Installation Challenges

TVM is notoriously difficult to install:
- ❌ Complex build dependencies (LLVM, CMake, CUDA toolkit)
- ❌ Platform-specific issues
- ❌ Poor pre-built wheel support
- ❌ Hours of debugging

### Better Alternatives

For learning GPU optimization and compilation concepts:

1. **Mega-Kernels Tutorial** (in this repo) ✅
   - Already working!
   - Teaches kernel fusion, memory optimization
   - Custom CUDA with PyTorch

2. **Triton** - GPU kernel language 🚀
   - Easy install: `pip install triton`
   - Write kernels in Python
   - Used in production (FlashAttention, xFormers)

3. **torch.compile()** - PyTorch 2.0 compiler
   - Built into PyTorch
   - Automatic fusion and optimization
   - No installation needed

4. **JAX/XLA** - Google's compiler
   - Easy install: `pip install jax`
   - Functional programming approach
   - Multi-platform support

**See [TVM_ALTERNATIVES.md](TVM_ALTERNATIVES.md) for details.**

---

## ❓ FAQ

### Q: Should I install TVM?

**A:** Probably not, unless you specifically need:
- Cross-compilation to exotic hardware (ARM, custom accelerators)
- Support for legacy frameworks (TensorFlow 1.x, MXNet)
- Academic research requiring TVM specifically

For learning and practical work, use the alternatives listed above.

### Q: What hardware does TVM support?

**A:**
- **GPUs:** NVIDIA (CUDA), AMD (ROCm), Intel, Arm Mali
- **CPUs:** x86 (Intel/AMD), ARM (Cortex-A/M)
- **Accelerators:** TPU (limited), custom accelerators
- **Mobile:** Android, iOS
- **Embedded:** Microcontrollers (microTVM)

### Q: Is TVM production-ready?

**A:** Yes! Used by AWS, Meta, AMD, and many others at scale. But installation complexity makes it challenging for individual developers.

---

## � Learning Resources in This Directory

| File | Description |
|------|-------------|
| **[LEARNING_GUIDE.md](LEARNING_GUIDE.md)** | Deep dive into TVM concepts, schedules, optimizations |
| **[REAL_WORLD_USES.md](REAL_WORLD_USES.md)** | Production use cases and case studies |
| **[TVM_ALTERNATIVES.md](TVM_ALTERNATIVES.md)** | Easier tools with similar capabilities |

## 🔗 External Resources

### Official Documentation
- **TVM Website:** https://tvm.apache.org/
- **Tutorials:** https://tvm.apache.org/docs/tutorial/
- **API Docs:** https://tvm.apache.org/docs/api/

### Community
- **Forum:** https://discuss.tvm.apache.org/
- **GitHub:** https://github.com/apache/tvm
- **Slack:** https://tvm.apache.org/community

### Papers
1. **TVM: An Automated End-to-End Optimizing Compiler for Deep Learning** (OSDI 2018)
2. **Ansor: Generating High-Performance Tensor Programs for Deep Learning** (OSDI 2020)

---

## 🎯 Recommended Next Steps

Instead of fighting TVM installation:

1. ✅ **Master mega-kernels** - Already working in this repo!
2. 🚀 **Learn Triton** - Modern GPU kernel language (easy install)
3. 📊 **Explore torch.compile()** - Built into PyTorch 2.0
4. 🔬 **Try JAX** - If interested in research/functional programming

### Practical Projects (Without TVM)

- ✨ Implement FlashAttention with Triton
- ✨ Optimize custom operators with CUDA
- ✨ Use torch.compile() for model optimization
- ✨ Build quantization kernels
- ✨ Explore kernel fusion patterns

---

## 📝 Summary

**What TVM offers:**
- ✅ Cross-platform compilation
- ✅ Auto-tuning for any hardware
- ✅ Operator fusion
- ✅ Graph-level optimizations

**Why it's challenging:**
- ❌ Difficult installation
- ❌ Complex dependencies
- ❌ Steep learning curve

**Better alternatives for most users:**
- **Mega-kernels** (this repo) - Learn GPU optimization fundamentals
- **Triton** - Modern, easy-to-use GPU kernel language  
- **torch.compile()** - Automatic optimization in PyTorch
- **JAX/XLA** - Functional approach with strong compilation

This directory serves as **conceptual reference** for understanding what TVM does and how it works, without requiring installation.
