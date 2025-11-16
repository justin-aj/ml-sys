# LLM Inference & Serving Optimization

This directory contains tutorials and guides for optimizing **Large Language Model (LLM) inference and serving**, focusing on production deployment, throughput, latency, and memory efficiency.

## 🎯 Overview

While the `distributed-training/` directory focuses on **training** large models across multiple GPUs, this directory focuses on **serving** trained models efficiently in production environments.

**Key Differences:**
- **Training**: Optimize for training throughput, gradient computation, parameter updates
- **Serving/Inference**: Optimize for request throughput, latency, memory efficiency, batching

---

## 📚 Tutorials

### 1. PagedAttention (vLLM) ⭐

**Tutorial**: [`pagedattention_tutorial/`](pagedattention_tutorial/)

**What it is**: Revolutionary memory management for LLM inference using virtual memory paging concepts for KV cache.

**Key Innovation**: Instead of pre-allocating contiguous memory for max sequence length, PagedAttention allocates fixed-size blocks on-demand, eliminating 85% memory waste.

**Performance**:
- 2-24× higher throughput vs traditional systems
- 6-8× larger batch sizes
- ~95% memory utilization (vs ~15% traditional)
- <5% computational overhead

**When to use**:
- ✅ High-throughput LLM API serving
- ✅ Variable-length sequences
- ✅ Parallel sampling (N outputs per prompt)
- ✅ Beam search
- ✅ Shared prompts across requests

**Files**:
- `README.md` - Comprehensive guide
- `CONCEPTS.md` - Mathematical deep dive
- `IMPLEMENTATION.md` - Code examples & CUDA kernels
- `COMPARISON.md` - vs TGI, FasterTransformer, Orca, FlashAttention
- `QUICKSTART.md` - Get started in 5 minutes
- 6 PNG visualizations

---

## 🎓 Learning Path

### Beginner
1. Start with **PagedAttention README.md** to understand the KV cache problem
2. Study visualizations to see traditional vs paged allocation
3. Read **QUICKSTART.md** to try vLLM

### Intermediate
1. Read **CONCEPTS.md** for mathematical foundations
2. Compare with other systems in **COMPARISON.md**
3. Understand block tables and memory sharing

### Advanced
1. Study **IMPLEMENTATION.md** for CUDA kernel details
2. Deploy vLLM in production
3. Tune block size and batch size for your workload

---

## 🔑 Key Concepts

### KV Cache Problem

During autoregressive generation, transformers cache attention keys (K) and values (V) to avoid recomputation:

```
Step 1: Compute K1, V1 for token 1
Step 2: Compute K2, V2 for token 2, reuse K1, V1
Step 3: Compute K3, V3 for token 3, reuse K1, K2, V1, V2
...
```

**Problem**: Traditional systems pre-allocate for max_seq_len (e.g., 2048 tokens), but:
- Average actual length: ~300 tokens
- Waste: 85% of allocated memory!

### PagedAttention Solution

Borrow virtual memory paging from operating systems:

```
Traditional:  [████████████░░░░░░░░░░░░░░░░░░░░] ← Must allocate 2048
PagedAttention: [████][████][██░░] ← Allocate only 2.5 blocks (40 tokens)
```

**Benefits**:
1. Allocate blocks on-demand
2. No fragmentation (all blocks same size)
3. Share blocks for common prefixes
4. Copy-on-write for diverging sequences

---

## 📊 Performance Comparison

**LLaMA-13B on A100 80GB:**

| System | Throughput | Batch Size | Memory Util |
|--------|-----------|------------|-------------|
| HuggingFace TGI | 0.8 req/s | 4 | ~15% |
| FasterTransformer | 1.2 req/s | 8 | ~20% |
| Orca (cont. batch) | 2.5 req/s | 16 | ~30% |
| **vLLM (PagedAttention)** | **19.3 req/s** | **64** | **~95%** |

**Speedup**: 24× over baseline!

---

## 🚀 Quick Start

### Install vLLM

```bash
pip install vllm
```

### Basic Usage

```python
from vllm import LLM, SamplingParams

# Initialize with PagedAttention (automatic)
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Generate
prompts = ["Hello, my name is"]
sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
outputs = llm.generate(prompts, sampling_params)

print(outputs[0].outputs[0].text)
```

**That's it!** PagedAttention is enabled by default in vLLM.

---

## 🔬 When to Use Each System

### Use vLLM (PagedAttention) if:
- High-throughput serving (APIs, chatbots)
- Variable sequence lengths
- Need parallel sampling or beam search
- Production deployment
- Want maximum GPU utilization

### Use Traditional Systems if:
- Single request at a time
- Research/experimentation (simpler setup)
- Fixed sequence lengths
- Small scale deployment

---

## 📖 Related Topics

### Complementary Techniques

PagedAttention works great with:

1. **FlashAttention** - Fast attention computation (3-4× faster kernels)
   - PagedAttention: Memory management
   - FlashAttention: Computation speed
   - **Combined**: 18-24× speedup!

2. **Quantization** (GPTQ, AWQ) - Reduce model size
   - 4-bit weights save 4× memory
   - More room for larger batches
   - Combined with PagedAttention: 10× more throughput

3. **Continuous Batching** - Dynamic batching
   - Built into vLLM
   - Add/remove requests dynamically
   - Better GPU utilization

4. **Tensor Parallelism** - Split across GPUs
   - For models too large for single GPU
   - vLLM supports tensor parallelism
   - PagedAttention works per-GPU

---

## 🏗️ Production Deployment

### Recommended Stack

```
┌─────────────────────────────────────┐
│    Load Balancer (nginx/HAProxy)    │
└─────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│      vLLM Server (PagedAttention)   │
│  - Continuous batching              │
│  - Memory sharing                   │
│  - High throughput                  │
└─────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│      GPU Cluster (A100/H100)        │
│  - Tensor parallelism               │
│  - Efficient KV cache               │
└─────────────────────────────────────┘
```

### Key Metrics to Monitor

- **Throughput**: requests/second
- **Latency**: P50, P95, P99
- **Memory utilization**: KV cache efficiency
- **Batch size**: concurrent requests
- **GPU utilization**: compute efficiency

---

## 📚 Additional Resources

### Papers
- **PagedAttention**: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)
- **FlashAttention**: "FlashAttention: Fast and Memory-Efficient Exact Attention" (NeurIPS 2022)
- **Orca**: "Orca: A Distributed Serving System for Transformer-Based Generative Models" (OSDI 2022)

### Code
- **vLLM**: https://github.com/vllm-project/vllm
- **Documentation**: https://docs.vllm.ai/

### Blogs
- vLLM Blog: https://blog.vllm.ai/
- UC Berkeley Sky Lab: https://sky.cs.berkeley.edu/

---

## 🎯 Summary

**PagedAttention** is a game-changer for LLM serving:

✅ **2-24× higher throughput** than traditional systems  
✅ **85% → 3% memory waste** reduction  
✅ **Production-ready** (used by major platforms)  
✅ **Easy to use** (few lines of code)  
✅ **Compatible** with FlashAttention, quantization, etc.

For high-throughput LLM serving, PagedAttention (via vLLM) is the current state-of-the-art.

---

*For distributed **training** strategies (ZeRO, Megatron, Alpa), see the `distributed-training/` directory.*
