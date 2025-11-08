# Triton Learning Guide: Deep Dive into GPU Kernel Fusion

## Table of Contents
1. [The Memory Problem](#the-memory-problem)
2. [Triton Programming Model](#triton-programming-model)
3. [Memory Hierarchy](#memory-hierarchy)
4. [Fusion Patterns](#fusion-patterns)
5. [Performance Analysis](#performance-analysis)
6. [Comparison with Alternatives](#comparison-with-alternatives)

---

## The Memory Problem

### Why Native PyTorch is Slow

Modern GPUs have **insane compute power** but are **starved for data**:

| Metric | NVIDIA V100 | NVIDIA A100 |
|--------|-------------|-------------|
| FP32 TFLOPS | 15.7 | 19.5 |
| Memory Bandwidth | 900 GB/s | 1555 GB/s |
| **Arithmetic Intensity Needed** | **17 ops/byte** | **12.5 ops/byte** |

**Problem:** Most operations do < 1 op/byte!

```python
# Example: element-wise operations
z = x + y  # Read 8 bytes (x, y), write 4 bytes, do 1 op
           # Arithmetic intensity = 1 op / 12 bytes = 0.08 ops/byte
           # GPU utilization: 0.08 / 17 = 0.5% of peak! 😱
```

### The Memory Hierarchy Gap

```
Access Time (relative):
Registers:       1x   ██
L1/Shared:      20x   ████████████████████
L2 Cache:      200x   ████████████████████████████████████████...
Global DRAM:   400x   ████████████████████████████████████████████████████████...
```

**PyTorch's problem:** Every operation goes to global DRAM.

**Triton's solution:** Keep data in registers/shared memory between operations.

---

## Triton Programming Model

### Core Concepts

#### 1. **Programs (Thread Blocks)**
```python
@triton.jit
def my_kernel(...):
    pid = tl.program_id(0)  # Which block am I?
    # Each program processes one chunk of data independently
```

- A "program" = CUDA thread block
- You launch N programs to process N chunks
- Each program runs on one Streaming Multiprocessor (SM)

#### 2. **Blocks of Data**
```python
BLOCK_SIZE = 1024  # Process 1024 elements at a time
offsets = tl.arange(0, BLOCK_SIZE)  # [0, 1, 2, ..., 1023]
```

- Load/store data in blocks, not element-by-element
- Block size is a critical tuning parameter
- Must be power of 2 for optimal performance

#### 3. **Pointer Arithmetic**
```python
# Load a block of data
x_ptrs = x_ptr + offsets
x = tl.load(x_ptrs, mask=offsets < n_elements)
```

- Work with pointers to memory locations
- Use masks for bounds checking
- Similar to C/C++ but type-safe

#### 4. **Automatic Parallelization**
```python
# This runs on ALL elements in the block simultaneously
y = x * 2 + 1  # Vectorized across the block!
```

- Operations on blocks are automatically parallelized
- No explicit loops over elements
- Compiler generates efficient SIMD code

---

## Memory Hierarchy

### 1. **Registers (Fastest)**
- **Latency:** ~1 cycle
- **Size:** 64 KB per SM (256 KB on A100)
- **Scope:** Private to each thread
- **Use case:** All intermediate computation

```python
# These stay in registers:
mean = tl.sum(x) / N
centered = x - mean
variance = tl.sum(centered * centered) / N
```

### 2. **Shared Memory / L1 Cache**
- **Latency:** ~20 cycles
- **Size:** 96 KB per SM (164 KB on A100)
- **Scope:** Shared within thread block
- **Use case:** Block-level reductions, data reuse

```python
# Triton automatically uses shared memory for reductions
total = tl.sum(x)  # Uses shared memory internally
```

### 3. **L2 Cache**
- **Latency:** ~200 cycles
- **Size:** 6 MB (V100), 40 MB (A100)
- **Scope:** Shared across entire GPU
- **Use case:** Automatic caching of recently accessed data

**Key insight:** If you load the same data twice in quick succession, it's often in L2!

### 4. **Global Memory (HBM/GDDR)**
- **Latency:** ~400 cycles
- **Size:** 32 GB (V100), 40-80 GB (A100)
- **Bandwidth:** 900 GB/s (V100), 1555 GB/s (A100)
- **Use case:** Initial load, final store

**Optimization goal:** Minimize global memory accesses.

---

## Fusion Patterns

### Pattern 1: Element-wise Fusion

**Operations that touch each element once:**

```python
# PyTorch (3 kernel launches)
x2 = x * 2
x2_plus_1 = x2 + 1
output = torch.exp(x2_plus_1)

# Triton (1 kernel launch)
@triton.jit
def fused_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)
    # All in registers! ↓
    x2 = x * 2
    x2_plus_1 = x2 + 1
    output = tl.exp(x2_plus_1)
    # ↑ All in registers!
    tl.store(out_ptr + offsets, output, mask=mask)
```

**Speedup:** ~2-3x (fewer launches + less memory)

### Pattern 2: Reduction Fusion

**Operations with reductions (sum, max, etc.):**

```python
# Softmax: max, exp, sum, divide
@triton.jit
def softmax_kernel(x_ptr, out_ptr, stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_ptr = x_ptr + row_idx * stride
    
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(row_ptr + cols, mask=cols < n_cols, other=-float('inf'))
    
    # All of this stays in registers!
    x_max = tl.max(x, axis=0)
    numerator = tl.exp(x - x_max)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator
    
    out_row_ptr = out_ptr + row_idx * stride
    tl.store(out_row_ptr + cols, output, mask=cols < n_cols)
```

**Speedup:** ~2-4x (eliminates temporary arrays)

### Pattern 3: Online Algorithms

**Compute in one pass what normally needs two:**

```python
# Layer Normalization: usually needs 2 passes (mean, then variance)
# But Welford's algorithm does it in 1 pass!

@triton.jit
def online_layernorm_kernel(...):
    # Load data once
    x = tl.load(x_ptr + offsets)
    
    # Welford's online variance algorithm
    mean = 0.0
    m2 = 0.0
    for i in range(n):
        delta = x[i] - mean
        mean += delta / (i + 1)
        m2 += delta * (x[i] - mean)
    variance = m2 / n
    
    # Normalize immediately (no second pass!)
    output = (x - mean) / tl.sqrt(variance + eps)
```

**Speedup:** Up to 2x (one pass instead of two)

---

## Performance Analysis

### Roofline Model

```
Performance (TFLOPS)
     │
Peak │ ████████████████ ← Compute bound (good!)
     │        ╱
     │       ╱
     │      ╱ ← Roofline
     │     ╱
     │    ╱
Memory   ╱ ← Memory bound (bad!)
bound │ ╱
     └─────────────────── Arithmetic Intensity (ops/byte)
```

**Goal:** Move right on the graph (higher arithmetic intensity) through fusion!

### Measuring Performance

```python
import time
import torch

def benchmark(fn, *args, num_runs=100):
    # Warmup
    for _ in range(10):
        fn(*args)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_runs):
        result = fn(*args)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / num_runs * 1000
    
    return ms, result

# Usage
pytorch_time, pytorch_out = benchmark(torch.softmax, x, dim=-1)
triton_time, triton_out = benchmark(triton_softmax, x)
speedup = pytorch_time / triton_time
```

### Key Metrics

1. **Time (ms):** Lower is better
2. **Throughput (GB/s):** Data processed per second
3. **TFLOPS:** Floating point operations per second
4. **Speedup:** Ratio of PyTorch time to Triton time

---

## Comparison with Alternatives

### Triton vs CUDA C++

| Aspect | CUDA C++ | Triton |
|--------|----------|--------|
| **Learning Curve** | Hard (memory management, sync) | Medium (Python-like) |
| **Development Speed** | Slow (compile, debug cycle) | Fast (instant compilation) |
| **Performance** | 100% (baseline) | 90-100% |
| **Portability** | NVIDIA only | NVIDIA (AMD/Intel planned) |
| **Auto-tuning** | Manual | Built-in |

**Use CUDA when:** Maximum control, reusable library code, extreme optimization
**Use Triton when:** Fast iteration, research, application-specific kernels

### Triton vs TVM

| Aspect | TVM | Triton |
|--------|-----|--------|
| **Installation** | ❌ Complex (build from source) | ✅ `pip install` |
| **Target** | Cross-platform ML models | NVIDIA GPU kernels |
| **Approach** | Auto-schedule + code generation | Explicit kernel programming |
| **Use Case** | Deploy models to many devices | Custom PyTorch operations |

**Use TVM when:** Cross-platform deployment, whole-model optimization
**Use Triton when:** Custom GPU kernels for PyTorch

### Triton vs torch.compile()

| Aspect | torch.compile() | Triton |
|--------|-----------------|--------|
| **Ease of Use** | ✅ Just add decorator | Medium (write kernels) |
| **Performance** | 1.2-2x speedup | 2-5x speedup |
| **Flexibility** | Limited to PyTorch ops | Arbitrary kernels |
| **Fusion** | Automatic | Manual |

**Use torch.compile() when:** Quick wins, standard PyTorch code
**Use Triton when:** Maximum performance, novel operations

---

## Best Practices

### 1. **Start Simple**
```python
# Don't:
@triton.jit
def mega_complex_kernel(...):
    # 500 lines of fused operations

# Do:
@triton.jit
def simple_fused_kernel(...):
    # 20 lines, one clear optimization
```

### 2. **Tune Block Sizes**
```python
# Use auto-tuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['n_elements'],
)
@triton.jit
def my_kernel(..., BLOCK_SIZE: tl.constexpr):
    ...
```

### 3. **Verify Correctness**
```python
# Always compare against PyTorch
pytorch_output = reference_implementation(x)
triton_output = triton_implementation(x)
assert torch.allclose(pytorch_output, triton_output, atol=1e-4)
```

### 4. **Profile Before Optimizing**
```python
# Use PyTorch profiler
with torch.profiler.profile() as prof:
    triton_output = triton_kernel(x)
print(prof.key_averages().table())
```

---

## Common Pitfalls

### 1. **Block Size Too Small**
```python
# Bad: Launches too many programs, overhead dominates
BLOCK_SIZE = 16  # Only 16 elements per kernel

# Good: Balance between parallelism and overhead
BLOCK_SIZE = 1024  # 1024 elements per kernel
```

### 2. **Not Using Masks**
```python
# Bad: Out-of-bounds access crashes!
x = tl.load(x_ptr + offsets)

# Good: Mask prevents out-of-bounds
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
```

### 3. **Unnecessary Global Memory Access**
```python
# Bad: Stores intermediate result
intermediate = x * 2
tl.store(temp_ptr + offsets, intermediate)  # ❌ Slow!
# ... later ...
intermediate = tl.load(temp_ptr + offsets)  # ❌ Slow!
result = intermediate + 1

# Good: Keep in registers
intermediate = x * 2  # Stays in registers
result = intermediate + 1  # Still in registers
tl.store(out_ptr + offsets, result)  # One write
```

---

## Next Steps

1. **Complete all tutorials** in order (simple_fusion → layer_norm → flash_attention)
2. **Read the Flash Attention paper** to see fusion enable new algorithms
3. **Profile your own PyTorch models** to find fusion opportunities
4. **Contribute to the Triton ecosystem** - it's open source!

**Resources:**
- [Triton Language Reference](https://triton-lang.org/main/python-api/triton.language.html)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Triton Tutorials (Official)](https://triton-lang.org/main/getting-started/tutorials/index.html)

---

**Remember:** Fusion isn't just about speed - it enables **new algorithms** that weren't possible with PyTorch's operation-by-operation model!
