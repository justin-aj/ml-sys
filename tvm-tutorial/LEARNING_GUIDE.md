# TVM Learning Guide

## Table of Contents
1. [What is TVM?](#what-is-tvm)
2. [Core Concepts](#core-concepts)
3. [Schedule Primitives](#schedule-primitives)
4. [GPU Optimization](#gpu-optimization)
5. [Auto-Tuning](#auto-tuning)
6. [Advanced Topics](#advanced-topics)

---

## What is TVM?

**TVM (Tensor Virtual Machine)** is an open-source deep learning compiler that optimizes ML models for any hardware backend.

### The Problem TVM Solves

**Traditional approach:**
```
PyTorch/TensorFlow → cuDNN/cuBLAS → NVIDIA GPU only
                   → MKL → Intel CPU only
                   → ...manual port for each hardware...
```

**TVM approach:**
```
PyTorch/TensorFlow/ONNX → TVM Compiler → Optimized code for ANY hardware
                                       → NVIDIA GPU, AMD GPU, ARM, x86, TPU, etc.
```

### Key Benefits

1. **Write Once, Run Anywhere**
   - Compile same model for GPU, CPU, mobile, edge devices
   - No need to rewrite kernels for each platform

2. **Automatic Optimization**
   - Auto-tuning finds best schedules for your hardware
   - Often faster than hand-written kernels

3. **Hardware Flexibility**
   - Not locked into NVIDIA ecosystem
   - Deploy to AMD, ARM, custom accelerators

4. **Operator Fusion**
   - Combines multiple operations into single kernels
   - Reduces memory bandwidth (like our mega kernels!)

---

## Core Concepts

### 1. Tensor Expressions (TE)

**Define WHAT to compute** (algorithm), not HOW to compute it (implementation).

```python
import tvm
from tvm import te

# Example: Element-wise addition
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute((n,), lambda i: A[i] + B[i], name="C")
```

**Key idea:** Declarative description of computation
- `te.placeholder`: Input tensor
- `te.compute`: Output computation
- `lambda i: ...`: How each element is computed

### 2. Schedule

**Define HOW to compute** (optimization strategy).

```python
# Create schedule
s = te.create_schedule(C.op)

# Apply optimizations (tiling, parallelization, etc.)
xo, xi = s[C].split(C.op.axis[0], factor=32)
s[C].parallel(xo)
s[C].vectorize(xi)
```

**Key idea:** Schedule separates algorithm from optimization
- Same computation can have many schedules
- Different schedules = different performance
- Auto-tuning searches for best schedule

### 3. Build and Execute

**Compile to machine code and run.**

```python
# Build function for target hardware
func = tvm.build(s, [A, B, C], target="cuda")  # or "llvm" for CPU

# Prepare data
dev = tvm.cuda(0)
a = tvm.nd.array(np.random.randn(1024).astype(np.float32), dev)
b = tvm.nd.array(np.random.randn(1024).astype(np.float32), dev)
c = tvm.nd.array(np.zeros(1024, dtype=np.float32), dev)

# Execute
func(a, b, c)
```

---

## Schedule Primitives

TVM provides many schedule primitives for optimization. Here are the most important ones:

### 1. Split - Divide Iteration Space

```python
# Split loop into outer and inner loops
# for i in range(0, 128):  →  for xo in range(0, 4):
#     ...                         for xi in range(0, 32):
#                                     i = xo * 32 + xi
xo, xi = s[C].split(x, factor=32)
```

**Use case:** Cache blocking, thread binding

### 2. Tile - 2D Split

```python
# Tile 2D space into blocks
xo, yo, xi, yi = s[C].tile(x, y, x_factor=32, y_factor=32)
```

**Use case:** Matrix operations, conv2d

### 3. Reorder - Change Loop Order

```python
# Change loop nesting order
s[C].reorder(xo, yo, ko, xi, yi, ki)
```

**Use case:** Improve cache locality, memory access patterns

### 4. Bind - Map to GPU Threads

```python
# Bind loops to GPU thread hierarchy
s[C].bind(xo, te.thread_axis("blockIdx.x"))
s[C].bind(xi, te.thread_axis("threadIdx.x"))
```

**Use case:** GPU parallelization

### 5. Vectorize - SIMD Instructions

```python
# Use vector instructions (SSE, AVX, NEON)
s[C].vectorize(xi)
```

**Use case:** CPU optimization, utilize SIMD units

### 6. Parallel - Multi-threading

```python
# Use multiple CPU cores
s[C].parallel(xo)
```

**Use case:** CPU multi-core parallelization

### 7. Unroll - Loop Unrolling

```python
# Unroll small loops
s[C].unroll(xi)
```

**Use case:** Reduce loop overhead, increase ILP

### 8. Cache Read/Write - Memory Hierarchy

```python
# Cache data in faster memory
AA = s.cache_read(A, "shared", [C])  # GPU shared memory
CC = s.cache_write(C, "local")       # GPU registers
```

**Use case:** GPU optimization, reduce global memory access

---

## GPU Optimization

### GPU Memory Hierarchy (V100)

```
Registers (fastest, ~KB per thread)
    ↕ 
Shared Memory (~96 KB per SM, shared by threads in block)
    ↕
L2 Cache (~6 MB, shared by all SMs)
    ↕
Global Memory (slowest, ~32 GB HBM2)
```

### Typical GPU Schedule Pattern

```python
# 1. Define computation
C = te.compute((M, N), lambda i, j: ..., name="C")
s = te.create_schedule(C.op)

# 2. Tiling for thread blocks
tile_x, tile_y = 64, 64
xo, xi = s[C].split(x, factor=tile_x)
yo, yi = s[C].split(y, factor=tile_y)

# 3. Cache in shared memory
AA = s.cache_read(A, "shared", [C])
BB = s.cache_read(B, "shared", [C])

# 4. Bind to GPU threads
s[C].bind(xo, te.thread_axis("blockIdx.x"))
s[C].bind(yo, te.thread_axis("blockIdx.y"))

# Further split for thread binding
xii, xiii = s[C].split(xi, factor=8)
yii, yiii = s[C].split(yi, factor=8)
s[C].bind(xii, te.thread_axis("threadIdx.x"))
s[C].bind(yii, te.thread_axis("threadIdx.y"))

# 5. Compute cached reads cooperatively
s[AA].compute_at(s[C], ko)  # Compute AA in the k loop
```

### Key GPU Optimization Principles

1. **Maximize Occupancy**
   - Use enough threads to hide memory latency
   - Typical: 256-1024 threads per block

2. **Coalesced Memory Access**
   - Adjacent threads access adjacent memory
   - Reorder loops to ensure coalescing

3. **Shared Memory Reuse**
   - Load data once, use multiple times
   - Reduces global memory bandwidth

4. **Register Blocking**
   - Keep frequently used data in registers
   - Accumulate results in registers

5. **Reduce Bank Conflicts**
   - Avoid threads accessing same shared memory bank
   - Pad arrays if necessary

---

## Auto-Tuning

### Why Auto-Tune?

**Manual scheduling is hard:**
- Huge search space (tile sizes, thread counts, etc.)
- Hardware-dependent (best schedule for V100 ≠ A100)
- Time-consuming to explore manually

**Auto-tuning automatically finds the best schedule!**

### AutoScheduler (Ansor)

TVM's automatic schedule generator.

```python
from tvm import auto_scheduler

# 1. Register workload
@auto_scheduler.register_workload
def matmul(M, N, K):
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k))
    return [A, B, C]

# 2. Create task
task = auto_scheduler.SearchTask(
    func=matmul,
    args=(1024, 1024, 1024),
    target="cuda"
)

# 3. Configure tuning
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=1000,  # Try 1000 schedules
    measure_callbacks=[auto_scheduler.RecordToFile("matmul.json")]
)

# 4. Run auto-tuning
task.tune(tune_option)

# 5. Apply best schedule
sch, args = task.apply_best("matmul.json")
```

### How AutoScheduler Works

1. **Sketch Generation**
   - Creates high-level schedule templates
   - Example: "Use tiling + shared memory + register blocking"

2. **Random Sampling**
   - Generates concrete schedules from sketches
   - Example: "Tile size = 32x32, 16 warps per block"

3. **Measurement**
   - Compiles and runs on actual hardware
   - Measures real performance (not a model!)

4. **Cost Model**
   - ML model predicts performance
   - Guides search to promising schedules
   - Avoids testing obviously bad schedules

5. **Evolutionary Search**
   - Keeps best schedules
   - Mutates/combines them
   - Iteratively improves

### Tuning Best Practices

1. **Start with more trials**
   - 100 trials: Quick test (~10 min)
   - 1000 trials: Good results (~1 hour)
   - 5000+ trials: Best results (several hours)

2. **Save tuning logs**
   - Reuse across runs (no need to re-tune)
   - Share with team

3. **Tune for your actual workload**
   - Use realistic input sizes
   - Tune for your target hardware

4. **Use early stopping**
   - Stop if no improvement for N trials
   - Saves time

---

## Advanced Topics

### 1. Relay - High-Level IR

TVM's graph-level IR for neural networks.

```python
from tvm import relay

# Import PyTorch model
model = torch.nn.Linear(128, 64)
input_shape = (1, 128)
input_data = torch.randn(input_shape)

# Convert to TVM Relay
traced = torch.jit.trace(model, input_data)
relay_mod, params = relay.frontend.from_pytorch(traced, [("input", input_shape)])

# Optimize graph
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(relay_mod, target="cuda", params=params)
```

**Benefits:**
- Operator fusion (combine multiple ops)
- Constant folding
- Dead code elimination
- Layout optimization

### 2. Tensor Cores

V100 has Tensor Cores for mixed precision (FP16).

```python
# Use tensor cores for matmul
with tvm.target.Target("cuda"):
    # Enable tensor core
    cfg.define_knob("use_tensorcore", [0, 1])
    
    if cfg["use_tensorcore"].val:
        # Use WMMA (Warp Matrix Multiply Accumulate)
        intrin = tvm.tir.TensorIntrin.register(
            "wmma_sync",
            wmma_sync_desc,
            wmma_sync_impl
        )
```

**Speedup:** 10-20x for mixed precision matmul!

### 3. Graph Optimizations

```python
# Operator fusion
# Before: conv2d → batch_norm → relu (3 kernels)
# After: fused_conv_bn_relu (1 kernel)

with relay.build_config(opt_level=3):
    # Automatic fusion
    graph, lib, params = relay.build(mod, target="cuda")
```

### 4. Cross-Compilation

```python
# Tune on V100, deploy on Jetson (ARM)
target = tvm.target.Target("cuda -model=jetson-nano")

# Or deploy on CPU
target = tvm.target.Target("llvm -mcpu=core-avx2")

# Or even WebAssembly!
target = tvm.target.Target("llvm -mtriple=wasm32-unknown-unknown")
```

---

## Comparison with Other Tools

### TVM vs CUDA

| Feature | CUDA | TVM |
|---------|------|-----|
| **Portability** | NVIDIA only | Any hardware |
| **Development** | Manual kernel writing | Auto-generated |
| **Optimization** | Manual tuning | Auto-tuning |
| **Learning curve** | Steep (C++, CUDA) | Moderate (Python) |
| **Peak performance** | Highest (if expert) | Close to peak (auto) |

**Use CUDA when:** You need absolute peak performance, NVIDIA-only deployment
**Use TVM when:** Multi-platform, rapid development, good-enough performance

### TVM vs Triton

| Feature | Triton | TVM |
|---------|--------|-----|
| **Language** | Python-like DSL | Python (TE) |
| **Target** | GPU only | Any hardware |
| **Auto-tuning** | Limited | Extensive |
| **Ease of use** | Easier | Moderate |
| **Maturity** | Newer | More mature |

**Use Triton when:** GPU-only, rapid kernel prototyping
**Use TVM when:** Multi-platform, production deployment

### TVM vs XLA (TensorFlow)

| Feature | XLA | TVM |
|---------|-----|-----|
| **Framework** | TensorFlow | Framework-agnostic |
| **Backends** | GPU, TPU, CPU | More backends |
| **Operator fusion** | ✓ | ✓ |
| **Auto-tuning** | Limited | Extensive |

**Use XLA when:** TensorFlow-only workflow
**Use TVM when:** Framework-agnostic, more backends

---

## Learning Resources

### Official Documentation
- **Website:** https://tvm.apache.org/
- **Tutorials:** https://tvm.apache.org/docs/tutorial/
- **Forum:** https://discuss.tvm.apache.org/

### Key Papers
1. **TVM: An Automated End-to-End Optimizing Compiler for Deep Learning** (OSDI 2018)
2. **Ansor: Generating High-Performance Tensor Programs for Deep Learning** (OSDI 2020)

### Practice Projects
1. Optimize convolution for your GPU
2. Compile a PyTorch ResNet model
3. Benchmark TVM vs PyTorch on custom ops
4. Deploy model to Raspberry Pi

---

## Summary

**TVM is a powerful deep learning compiler that:**
- ✅ Compiles models for any hardware
- ✅ Auto-tunes for optimal performance
- ✅ Provides fine-grained control when needed
- ✅ Integrates with PyTorch, TensorFlow, ONNX

**Key concepts:**
1. **Tensor Expressions** - WHAT to compute
2. **Schedule** - HOW to compute
3. **Auto-tuning** - Find best schedule automatically
4. **Relay** - High-level graph optimizations

**When to use TVM:**
- Multi-platform deployment
- Custom operators not in libraries
- Performance-critical inference
- Research on ML compilers

**Learning Path:** Start with this guide, then REAL_WORLD_USES.md, then explore alternatives in TVM_ALTERNATIVES.md!
