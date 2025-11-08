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
