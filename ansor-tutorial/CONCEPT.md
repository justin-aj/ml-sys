# Ansor: Learned Auto-Scheduling for GPUs

> **⚠️ NOTE:** This is a **concept-only** document for educational reference.  
> **No hands-on tutorial** is provided due to TVM installation complexity.  
> This document explains what Ansor is, why it's unique, and how it compares to other tools like Triton.

---

## 🎯 The Core Idea

**Ansor is the ONLY tool that does ALL of this:**

1. ✅ **Loop reordering search** → Automatically explores loop permutations
2. ✅ **Tiling factor exploration** → Searches tile/block sizes, vectorization widths, unrolling factors
3. ✅ **Cross-device cost models** → Learns predictive models of kernel runtime for different GPUs/CPUs
4. ✅ **Learned scheduling** → Trains ML models (gradient-boosted trees) to guide schedule search for unseen workloads

---

## 🤔 Why Not Other Tools?

| Tool | What It Does | What It Lacks |
|------|--------------|---------------|
| **Triton** | JIT compiler + manual kernel authoring, fuses operations | ❌ No schedule search<br>❌ No learned cost models<br>❌ Manual optimization |
| **TorchInductor** | Heuristic-based fusion and tiling | ❌ No learned cost model<br>❌ Fixed heuristics, not adaptive |
| **TensorRT** | Autotunes GEMM/convolution kernels | ❌ Limited to supported layers<br>❌ Cost model not exposed/generalized |
| **Hidet / MLIR** | Partial search or polyhedral optimization | ❌ Not ML-guided<br>❌ Limited cross-device learning |

**Bottom Line:** If your goal is **full learned auto-scheduling with cost models and loop exploration across devices**, **TVM + Ansor (or Meta-Scheduler)** is the only open-source system that does this end-to-end.

---

## 🧠 What Makes Ansor Unique?

### The Problem Ansor Solves

When optimizing a GPU kernel, you have a **MASSIVE** search space:

```
Loop Order Choices:
  matmul: [i, j, k] vs [k, i, j] vs [i, k, j] vs ...
  → 3! = 6 permutations for just 3 loops
  → Real kernels have 5-10 loops = millions of permutations

Tiling Choices:
  Tile size for i: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
  Tile size for j: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
  Tile size for k: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
  → 11 × 11 × 11 = 1,331 combinations just for tiling!

Vectorization/Unrolling:
  Which loops to vectorize? (4, 8, 16 wide?)
  Which loops to unroll? (2, 4, 8 times?)
  → Hundreds more combinations

Hardware Mapping:
  Which loop maps to threadIdx.x vs blockIdx.x?
  How many threads per block?
  → Device-specific choices

Total Search Space: 10^10 to 10^15 possible schedules! 🤯
```

### Ansor's Solution: Machine Learning

Instead of trying random schedules or using fixed heuristics, Ansor:

1. **Samples** a small subset of schedules (e.g., 1000 out of 10^12)
2. **Measures** their actual runtime on your GPU
3. **Learns** a cost model (gradient-boosted tree) that predicts performance
4. **Explores** promising regions of the search space using the learned model
5. **Transfers** knowledge across similar workloads and devices

**Result:** Find near-optimal schedules in minutes instead of days/weeks of manual tuning!

---

## 🔬 How Ansor Works (High-Level)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INPUT: Your Computation (e.g., MatMul, Conv2D)          │
│    Written in TVM's compute description language           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. SKETCH GENERATION                                        │
│    Ansor generates "sketch templates":                      │
│    - Loop structures (nested loops, tiling patterns)        │
│    - Parallelization strategies (thread/block mapping)      │
│    - Memory hierarchy usage (shared/global/register)        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. EVOLUTIONARY SEARCH                                      │
│    For each sketch:                                         │
│    - Sample random schedules (tile sizes, loop orders)      │
│    - Run on GPU and measure time                            │
│    - Train cost model (XGBoost) to predict performance      │
│    - Use model to guide search toward better schedules      │
│    - Iterate 100-10,000 times                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. BEST SCHEDULE FOUND                                      │
│    Output: Optimized CUDA/OpenCL kernel code                │
│    Performance: Often matches or beats hand-tuned kernels!  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎓 What Ansor Can Do (Conceptual Overview)

### Capability 1: **Simple MatMul Auto-Tuning**
- Define a matrix multiplication in TVM compute language
- Run Ansor auto-tuning (1000+ trials) to explore schedules
- Automatically find loop orders, tile sizes, parallelization strategies
- **Expected Result:** 80-95% of cuBLAS performance (fully auto-generated!)

### Capability 2: **Conv2D Schedule Exploration**
- Define a 2D convolution workload
- Ansor explores different schedule strategies:
  - Spatial tiling vs channel tiling
  - Shared memory usage patterns
  - Vectorization and unrolling factors
- Chooses different strategies for different input shapes automatically
- **Expected Result:** Matches or beats cuDNN for many configurations

### Capability 3: **Cross-Device Transfer Learning**
- Tune a kernel on GPU A (e.g., V100)
- Transfer the learned cost model to GPU B (e.g., A100)
- Ansor adapts with minimal re-tuning using transfer learning
- **Expected Result:** 50-80% faster convergence vs tuning from scratch

### Capability 4: **Custom Operator Auto-Tuning**
- Define custom fused operations (e.g., LayerNorm + GELU)
- Ansor finds optimal schedule without manual kernel engineering
- Competitive with hand-written CUDA/Triton kernels
- **Expected Result:** Competitive performance in hours vs weeks of manual work

---

## 🚀 Why This Matters for Real-World ML

### Problem: Every Model Has Unique Kernels

Modern ML models have:
- **Unique shapes:** GPT-4 (vocab=100k), LLaMA (vocab=32k), different batch sizes
- **Custom operators:** FlashAttention, PagedAttention, custom normalizations
- **Multiple devices:** Train on A100, deploy on H100, inference on T4
- **Evolving architectures:** New models appear weekly!

**Manual tuning doesn't scale** → You can't hand-write CUDA kernels for every combination!

### Ansor's Value Proposition

```
Traditional Approach:
  New model → Write CUDA kernel → Tune for weeks → Deploy
  New GPU → Re-tune all kernels → Weeks more work
  New operator → Start from scratch → More weeks
  Total: MONTHS of expert engineering time

Ansor Approach:
  New model → Write TVM compute → Auto-tune overnight → Deploy
  New GPU → Transfer learning → Re-tune hours → Deploy
  New operator → Write TVM compute → Auto-tune → Deploy
  Total: DAYS of work, mostly automated
```

**Real Impact:**
- **OctoML** (founded by TVM creators): Uses Ansor to optimize models for 100+ device types
- **AWS**: Uses TVM/Ansor in SageMaker Neo for edge deployment
- **Meta**: Uses auto-scheduling for PyTorch model optimization

---

## 📊 Expected Performance

Based on TVM/Ansor papers and real-world usage:

| Workload | Ansor vs Hand-Tuned | Ansor vs PyTorch |
|----------|---------------------|------------------|
| **MatMul (1024×1024)** | 90-95% of cuBLAS | ~1.0x (cuBLAS is gold standard) |
| **Conv2D (ResNet)** | 85-100% of cuDNN | 0.9-1.1x (competitive!) |
| **Custom Fusions** | Often better! | 1.2-2.0x (PyTorch can't fuse) |
| **Sparse/Irregular** | 80-120% of hand-tuned | 1.5-3.0x (no native PyTorch support) |

**Key Insight:** Ansor shines for:
1. Custom operators where hand-tuning is expensive
2. New hardware where libraries aren't optimized yet
3. Fused operations that frameworks can't handle
4. Rapid prototyping (hours vs weeks of kernel engineering)

---

## 🛠️ What Would Be Needed (Installation Reality Check)

**Why no hands-on tutorial?** TVM installation is complex:

```bash
# TVM installation challenges:
- Requires building from source (no simple pip install)
- Complex dependencies: LLVM, CUDA toolkit, CMake, C++ compiler
- Platform-specific issues (especially on Windows)
- Build time: 30-60 minutes
- Frequent compatibility issues between TVM/LLVM/CUDA versions
```

**Conceptual file structure** (if TVM were easy to install):
```
ansor-tutorial/
├── CONCEPT.md                   # This file (educational reference)
├── INSTALLATION.md              # Would detail TVM setup (complex!)
├── example_matmul.py            # Would show auto-tuning workflow
├── example_conv2d.py            # Would show schedule exploration
├── example_transfer.py          # Would show cross-device learning
├── example_custom.py            # Would show custom operator tuning
└── COMPARISON.md                # Ansor vs Triton vs PyTorch
```

**Reality:** Triton tutorial is much more practical for learning GPU optimization!

---

## 🎯 Key Takeaways (Educational)

After understanding this concept, you should know:

1. ✅ **Understand** the auto-scheduling search space is MASSIVE (10^10+ possible schedules)
2. ✅ **Know** Ansor uses machine learning (XGBoost) to navigate this space efficiently
3. ✅ **Recognize** Ansor is the only tool that does learned auto-scheduling end-to-end
4. ✅ **Compare** auto-tuning (Ansor) vs manual optimization (Triton) trade-offs
5. ✅ **Appreciate** transfer learning across devices (V100 → A100)
6. ✅ **Decide when** to use Ansor vs other tools in real-world scenarios

**Practical advice:** For hands-on GPU optimization learning, use the **Triton tutorial** instead!

---

## 🔮 The Future: Meta-Scheduler

**Note:** TVM recently evolved Ansor into **Meta-Scheduler**, which adds:
- Better sketch generation (more diverse search space)
- Improved cost models (neural networks + XGBoost)
- Dynamic shape support (variable batch sizes)
- Better integration with PyTorch/JAX

We'll cover **both** Ansor (simpler, educational) and Meta-Scheduler (cutting-edge) in this tutorial.

---

## 🤝 How This Complements the Triton Tutorial

| Aspect | Triton Tutorial | Ansor Tutorial |
|--------|----------------|----------------|
| **Approach** | Manual kernel engineering | Automated schedule search |
| **Control** | Full control over every detail | High-level compute definition |
| **Speed to solution** | Write kernel → profile → tune (hours-days) | Define compute → auto-tune (minutes-hours) |
| **Performance ceiling** | 95-100% of hand-tuned CUDA | 80-95% of hand-tuned (but automated!) |
| **Best for** | Critical kernels, learning GPU programming | Rapid prototyping, many operators, new hardware |
| **Learning curve** | Moderate (need GPU understanding) | Steeper (need TVM + ML concepts) |

**Use them together!**
- Prototype with Ansor to find good schedules quickly
- Use Triton for final 5-10% optimization of critical kernels
- Or just use Ansor for everything if 90% performance is good enough!

---

## 🎬 What to Do Instead

**Want hands-on GPU optimization?**

1. ✅ **Complete the Triton tutorial** (in `../triton-tutorial/`)
   - Practical, pip-installable, runs on your GPU
   - Learn fusion, memory optimization, real kernel engineering
   - See actual 1.3-1.5x speedups on V100

2. 📚 **Read Ansor papers** (conceptual understanding)
   - [Ansor: Generating High-Performance Tensor Programs](https://arxiv.org/abs/2006.06762)
   - [TVM: An Automated End-to-End Optimizing Compiler](https://arxiv.org/abs/1802.04799)

3. 🏢 **Use production tools** (when TVM is already set up)
   - OctoML platform (TVM as-a-service, no installation needed)
   - AWS SageMaker Neo (uses TVM under the hood)
   - Pre-built TVM Docker images (avoid building from source)

**Bottom line:** Understand Ansor conceptually, use Triton practically!

---

*"The best GPU kernel is the one you don't have to write yourself."* — TVM Philosophy

---

## 📚 Additional Resources

**Papers:**
- [Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://arxiv.org/abs/2006.06762)
- [Learning to Optimize Tensor Programs](https://arxiv.org/abs/1805.08166)
- [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/abs/1802.04799)

**Production Systems Using Ansor/TVM:**
- **OctoML:** TVM-as-a-service platform for model optimization
- **AWS SageMaker Neo:** Model optimization for edge deployment
- **Meta/Facebook:** PyTorch model optimization backend
- **AMD:** ROCm ML framework optimization

**When You Might Actually Use Ansor:**
- Your company already has TVM infrastructure set up
- Optimizing for exotic/new hardware (RISC-V, custom accelerators)
- Need cross-device optimization (100+ device types)
- Research projects exploring auto-scheduling algorithms
- You have ML engineers who can maintain TVM builds

---

*This document serves as educational reference for understanding the landscape of GPU optimization tools. For practical, hands-on learning, see the Triton tutorial in `../triton-tutorial/`.*
