# ML Systems Projects

A collection of machine learning systems and GPU optimization projects.

---

## 📂 Projects

### [mega-kernels/](./mega-kernels/)

**Learn CUDA Mega Kernels with hands-on examples**

A comprehensive tutorial demonstrating kernel fusion optimization techniques for NVIDIA GPUs. Learn how to achieve 1.6x - 1.9x speedups by eliminating memory bandwidth bottlenecks.

#### What are Mega Kernels?

Mega kernels (also called "kernel fusion") combine multiple GPU operations into a single kernel to:
- **Reduce memory bandwidth** by 50% (avoid intermediate global memory roundtrips)
- **Eliminate kernel launch overhead** (1 launch instead of multiple)
- **Keep data in registers** between operations (much faster than global memory)

#### Key Results (Tesla V100-SXM2-32GB)

| Tensor Size | Standard | Mega Kernel | Speedup |
|-------------|----------|-------------|---------|
| 1K elements | 0.0156 ms | 0.0091 ms | **1.72x** |
| 1M elements | 0.0217 ms | 0.0136 ms | **1.59x** |
| 16M elements | 0.3351 ms | 0.1752 ms | **1.91x** |

**Example:** GELU activation + scaling fused into one kernel achieves near-theoretical 2x speedup on large tensors.

#### What's Included

- 📚 **tutorial_mega_kernel.py** - Educational CUDA kernel fusion example
- 🎯 **simple_demo.py** - PyTorch-only demonstration (no custom CUDA)
- 📖 **LEARNING_GUIDE.md** - Step-by-step conceptual tutorial
- 📊 **BENCHMARK_RESULTS.md** - Detailed V100 performance analysis
- 🌍 **REAL_WORLD_USES.md** - Production applications (FlashAttention, Megatron-LM, etc.)

#### Quick Start

```bash
cd mega-kernels
pip install -r requirements.txt
python tutorial_mega_kernel.py
```

**Requirements:** PyTorch >= 2.0.0 with CUDA support

#### Core Concept

```python
# Standard approach (2 kernels, 2 memory roundtrips)
temp = gelu(input)      # GPU kernel 1: read input, write temp to global memory
output = temp * scale   # GPU kernel 2: read temp, write output to global memory

# Mega kernel (1 kernel, 1 memory roundtrip)
output = fused_gelu_scale(input, scale)  # read input, compute both ops in registers, write output
```

**Result:** 50% less memory traffic → up to 1.91x faster on V100

#### Real-World Impact

This technique is used in:
- **FlashAttention** - Attention mechanism optimization (2-4x faster)
- **Megatron-LM** - Training 100B+ parameter models
- **xFormers** - Memory-efficient transformers
- **TensorRT** - Production inference optimization
- **torch.compile()** - PyTorch 2.0 automatic fusion

#### Learn More

See the [mega-kernels directory](./mega-kernels/) for complete tutorials, benchmarks, and real-world examples.

---

### [tvm-tutorial/](./tvm-tutorial/)

**Learn TVM: The Deep Learning Compiler**

A comprehensive tutorial on Apache TVM - an open-source compiler that optimizes ML models for any hardware. Learn how to achieve cross-platform deployment with automatic performance tuning.

#### What is TVM?

TVM (Tensor Virtual Machine) is a compiler framework that:
- **Compiles models for ANY hardware** (NVIDIA, AMD, ARM, TPU, custom accelerators)
- **Auto-tunes schedules** to match/beat hand-written kernels
- **Enables operator fusion** (like mega kernels) automatically
- **Supports all frameworks** (PyTorch, TensorFlow, ONNX)

Think: "Write once, deploy everywhere" - the ultimate ML deployment tool.

#### Key Concepts Demonstrated

| Concept | Description | Benefit |
|---------|-------------|---------|
| **Tensor Expressions** | Define WHAT to compute | Algorithm-implementation separation |
| **Schedules** | Define HOW to compute | Performance tuning without changing logic |
| **Auto-Tuning** | Automatic schedule search | Find optimal performance automatically |
| **Operator Fusion** | Combine multiple ops | Reduce memory bandwidth (like mega kernels!) |

#### What's Included

- 🎯 **simple_intro.py** - CPU tutorial comparing PyTorch vs TVM (no GPU needed)
- 🚀 **gpu_optimization.py** - GPU optimization on V100 (manual schedules)
- 🤖 **auto_tuning.py** - Automatic schedule search with AutoScheduler
- 📖 **LEARNING_GUIDE.md** - Deep dive: schedules, primitives, GPU optimization
- 🌍 **REAL_WORLD_USES.md** - Production deployments (AWS, Meta, AMD)

#### Quick Start

```bash
cd tvm-tutorial
pip install -r requirements.txt

# Start with CPU tutorial (no GPU required)
python simple_intro.py

# Then try GPU optimization (if you have CUDA)
python gpu_optimization.py

# Finally, let TVM auto-tune (takes 10-60 min)
python auto_tuning.py
```

**Requirements:** Python 3.8+, PyTorch >= 2.0.0, apache-tvm >= 0.14.0

#### Example: CPU Optimization

```python
# Define computation (WHAT)
A = te.placeholder((n, n), name="A")
B = te.placeholder((n, n), name="B")
C = te.compute((n, n), lambda i, j: A[i, j] + B[i, j])

# Basic schedule (slow)
s = te.create_schedule(C.op)

# Optimized schedule (HOW) - fast!
xo, xi = s[C].split(C.op.axis[0], factor=32)  # Tiling
s[C].parallel(xo)     # Multi-core
s[C].vectorize(xi)    # SIMD instructions

# Build and run
func = tvm.build(s, [A, B, C], target="llvm")
```

**Result:** 2-5x faster than naive implementation through schedule optimization!

#### Real-World Impact

TVM powers production systems at:
- **AWS** - Inferentia chips and SageMaker Neo
- **Meta** - Transformer inference (billions of requests/day)
- **AMD** - ROCm stack for GPU deployment
- **OctoML** - Commercial ML deployment platform
- **Arm** - Mobile and edge AI deployment

#### Key Learning Outcomes

After completing this tutorial, you'll understand:

✅ **Separation of concerns** - Algorithm (TE) vs Optimization (Schedule)  
✅ **Schedule primitives** - split, tile, reorder, vectorize, parallel  
✅ **GPU optimization** - thread binding, shared memory, register blocking  
✅ **Auto-tuning** - Let ML find optimal schedules automatically  
✅ **Cross-compilation** - Same code, multiple hardware targets  
✅ **Operator fusion** - Memory bandwidth optimization  

#### TVM vs Other Tools

| Tool | Portability | Auto-Tuning | Ease of Use | Best For |
|------|-------------|-------------|-------------|----------|
| **CUDA** | NVIDIA only | Manual | Hard | Peak GPU performance |
| **Triton** | GPU only | Limited | Easy | Rapid GPU prototyping |
| **XLA** | GPU/TPU/CPU | Limited | Easy | TensorFlow ecosystem |
| **TVM** | All hardware | Excellent | Moderate | **Multi-platform deployment** |

#### Performance Examples

Typical speedups over framework defaults:

| Model | Framework | TVM | Speedup |
|-------|-----------|-----|---------|
| ResNet-50 | PyTorch | TVM (auto-tuned) | 1.8x |
| BERT-Base | PyTorch | TVM (auto-tuned) | 2.7x |
| Custom Fusion | PyTorch (2 kernels) | TVM (1 kernel) | 2.0x |

*Tested on V100 GPU, batch size 1, FP32 precision*

#### When to Use TVM

✅ **Use TVM when:**
- Deploying to multiple hardware platforms
- Need automatic optimization for custom ops
- Targeting non-NVIDIA hardware (AMD, ARM, etc.)
- Want framework-agnostic deployment (PyTorch + TF support)

❌ **Don't use TVM when:**
- NVIDIA-only deployment with standard ops (cuBLAS/cuDNN is simpler)
- Quick prototyping (use PyTorch/JAX directly)
- Need training optimization (TVM focuses on inference)

#### Learn More

See the [tvm-tutorial directory](./tvm-tutorial/) for:
- Complete code examples with detailed comments
- Step-by-step learning guide
- Auto-tuning best practices
- Real-world production use cases

---

## 🚀 Getting Started

Clone this repository:
```bash
git clone https://github.com/justin-aj/ml-sys.git
cd ml-sys
```

Navigate to individual project folders for specific setup instructions.

---

## 📋 Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (tested on Tesla V100)
- PyTorch >= 2.0.0

---

## 📄 License

MIT License - See individual project folders for details.

---

## 🤝 Contributing

Feel free to explore, learn, and adapt these examples for your own projects!
