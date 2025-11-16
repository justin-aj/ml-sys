# PagedAttention: Comprehensive Comparison

## Table of Contents
- [PagedAttention vs Traditional KV Cache](#pagedattention-vs-traditional-kv-cache)
- [PagedAttention vs Other Serving Systems](#pagedattention-vs-other-serving-systems)
- [PagedAttention vs Attention Optimizations](#pagedattention-vs-attention-optimizations)
- [PagedAttention vs Memory Techniques](#pagedattention-vs-memory-techniques)
- [Hybrid Approaches](#hybrid-approaches)
- [Decision Framework](#decision-framework)

---

## PagedAttention vs Traditional KV Cache

### Memory Management Comparison

| Aspect | Traditional | PagedAttention | Winner |
|--------|------------|----------------|---------|
| **Allocation** | Pre-allocate max_seq_len | On-demand blocks | ✅ Paged |
| **Memory waste** | 85% (avg) | 3% (avg) | ✅ Paged |
| **Fragmentation** | Internal + External | Minimal (internal only) | ✅ Paged |
| **Sharing** | Not supported | Copy-on-write | ✅ Paged |
| **Complexity** | Simple | Moderate (block tables) | ❌ Trad |
| **Overhead** | 0% | 3-5% | ❌ Trad |

### Performance Metrics

**LLaMA-13B on A100 80GB:**

| Metric | Traditional | PagedAttention | Improvement |
|--------|------------|----------------|-------------|
| Memory utilization | 14.6% | 97.4% | **6.7×** |
| Max batch size | 8 | 52 | **6.5×** |
| Throughput (req/s) | 1.2 | 14.5 | **12×** |
| Latency (P99) | 3.2s | 0.8s | **4×** |

### Code Complexity

**Traditional:**
```python
# Simple contiguous allocation
kv_cache = torch.zeros(batch_size, max_seq_len, num_layers, 2, num_heads, head_dim)

# Append new token
kv_cache[:, seq_len] = new_kv

# Attention
output = attention(query, kv_cache[:, :seq_len])
```

**PagedAttention:**
```python
# Block-based allocation (more complex)
block_manager = BlockSpaceManager(block_size=16)
block_tables = {}

# Allocate blocks on-demand
if seq_len % block_size == 0:
    new_block = block_manager.allocate()
    block_tables[seq_id].append(new_block)

# Attention with block lookup
output = paged_attention(query, kv_cache_blocks, block_tables[seq_id])
```

**Verdict**: PagedAttention is more complex but worth it for production serving.

---

## PagedAttention vs Other Serving Systems

### 1. vLLM (PagedAttention) vs HuggingFace TGI

**HuggingFace Text Generation Inference (TGI):**
- Uses traditional contiguous KV cache
- Static batching (batch completes when slowest finishes)
- No memory sharing

**Comparison (LLaMA-13B, A100 80GB):**

| Metric | TGI | vLLM | Speedup |
|--------|-----|------|---------|
| Throughput | 0.8 req/s | 19.3 req/s | **24×** |
| Max batch size | 4 | 64 | **16×** |
| Memory util | ~15% | ~95% | **6.3×** |
| Latency (avg) | 2.5s | 0.6s | **4.2×** |

**Why vLLM wins:**
- ✅ PagedAttention eliminates waste
- ✅ Continuous batching keeps GPU busy
- ✅ Larger batches improve throughput
- ✅ Memory sharing for parallel sampling

### 2. vLLM vs FasterTransformer (NVIDIA)

**FasterTransformer:**
- Optimized CUDA kernels for transformers
- Better than TGI but still uses contiguous cache
- Limited batching flexibility

**Comparison (LLaMA-13B, A100 80GB):**

| Metric | FasterTransformer | vLLM | Speedup |
|--------|-------------------|------|---------|
| Throughput | 1.2 req/s | 19.3 req/s | **16×** |
| Max batch size | 8 | 64 | **8×** |
| Memory util | ~20% | ~95% | **4.8×** |

**Why vLLM wins:**
- ✅ Better memory management (PagedAttention)
- ✅ Continuous batching
- ✅ Can combine PagedAttention + FasterTransformer kernels

### 3. vLLM vs Orca (Microsoft)

**Orca:**
- Introduced **continuous batching** (iteration-level scheduling)
- Still uses contiguous KV cache
- Better than static batching

**Comparison (GPT-3 175B, 8×A100 80GB):**

| Metric | Orca | vLLM | Speedup |
|--------|------|------|---------|
| Throughput | 2.5 req/s | 19.3 req/s | **7.7×** |
| Max batch size | 16 | 64 | **4×** |
| Memory util | ~30% | ~95% | **3.2×** |

**Analysis:**
- Orca's continuous batching: **3×** speedup over static batching
- vLLM's PagedAttention: **6×** speedup over contiguous cache
- **Combined effect**: ~18× over baseline (TGI)

**vLLM = Continuous Batching + PagedAttention**

### 4. vLLM vs Text Generation WebUI

**Text Generation WebUI:**
- Simple web interface for LLMs
- No optimization (traditional PyTorch/Transformers)
- Single request at a time

**Comparison:**

| Metric | WebUI | vLLM | Speedup |
|--------|-------|------|---------|
| Throughput | 0.3 req/s | 19.3 req/s | **64×** |
| Max concurrent | 1 | 64+ | **64×** |
| GPU utilization | ~40% | ~95% | **2.4×** |

**Why such huge gap:**
- No batching vs advanced batching
- Naive memory vs PagedAttention
- Not production-optimized vs highly optimized

---

## PagedAttention vs Attention Optimizations

### 1. PagedAttention vs FlashAttention

**FlashAttention (Dao et al., 2022):**
- Optimizes attention **computation** (not memory management)
- Uses tiling to reduce HBM accesses
- 2-4× faster attention kernel

**Key difference:**
- **FlashAttention**: Makes attention computation faster
- **PagedAttention**: Makes KV cache memory efficient

**They are COMPLEMENTARY!**

```
Traditional Attention:
    Computation: Slow (standard GEMM)
    Memory: Wasteful (contiguous cache)
    Throughput: 1×

FlashAttention:
    Computation: Fast (tiled, IO-aware)     ✅ 3× faster
    Memory: Wasteful (contiguous cache)     ❌ Same waste
    Throughput: 3×

PagedAttention:
    Computation: Slow (standard GEMM)       ❌ Same speed
    Memory: Efficient (block-based)         ✅ 6× better
    Throughput: 6×

FlashAttention + PagedAttention:
    Computation: Fast (tiled, IO-aware)     ✅ 3× faster
    Memory: Efficient (block-based)         ✅ 6× better
    Throughput: 18× (3 × 6)                 🚀 BEST!
```

**vLLM actually uses FlashAttention + PagedAttention!**

### 2. PagedAttention vs Flash-Decoding

**Flash-Decoding (Tri Dao, 2023):**
- Further optimizes FlashAttention for **decoding** (single token generation)
- Parallelizes over sequence length dimension
- ~2× faster than FlashAttention for long sequences

**Comparison:**

| Technique | Target | Benefit | Compatible with Paged? |
|-----------|--------|---------|------------------------|
| FlashAttention | Prefill | 3-4× faster | ✅ Yes |
| Flash-Decoding | Decode | 2× faster | ✅ Yes |
| PagedAttention | Memory | 6× better | ✅ Yes (orthogonal) |

**Best combination:** FlashAttention + Flash-Decoding + PagedAttention

### 3. PagedAttention vs Multi-Query Attention (MQA)

**Multi-Query Attention (Shazeer, 2019):**
- Share K, V across all heads (instead of separate per head)
- Reduces KV cache size: $h \times$ smaller (where $h$ = num heads)
- Faster inference but slightly lower quality

**KV cache size comparison (LLaMA-13B, 40 heads):**

| Method | KV cache per token | Reduction |
|--------|-------------------|-----------|
| Standard (GQA) | 819 KB | 1× |
| MQA | 20 KB | **40×** |

**Comparison:**

| Aspect | MQA | PagedAttention | Winner |
|--------|-----|----------------|---------|
| **Memory reduction** | 40× (fewer KV) | 6× (less waste) | ✅ MQA |
| **Quality** | Slightly lower | No change | ✅ Paged |
| **Compatibility** | Requires model change | Works with any model | ✅ Paged |
| **Batch size** | Larger (less mem per seq) | Larger (less waste) | 🤝 Both |

**Can combine:** MQA + PagedAttention = 240× memory improvement!

---

## PagedAttention vs Memory Techniques

### 1. PagedAttention vs Quantization (GPTQ, AWQ)

**Quantization (GPTQ, AWQ):**
- Reduce model weights from FP16 to INT4/INT8
- 2-4× memory savings on weights
- Can also quantize KV cache

**Comparison:**

| Technique | Target | Savings | Quality Loss |
|-----------|--------|---------|--------------|
| GPTQ/AWQ | Weights | 2-4× | Minimal (~1%) |
| KV cache quant (INT8) | KV cache | 2× | Minimal |
| PagedAttention | KV cache | 6× (waste) | None |

**Combined effect (for 13B model on 40GB A100):**

```
Original:
    Weights: 26 GB (FP16)
    KV cache: 14 GB (traditional, batch=8)
    Total: 40 GB

Quantization only (INT4 weights, INT8 KV):
    Weights: 6.5 GB
    KV cache: 7 GB
    Total: 13.5 GB
    Batch size: ~24

PagedAttention only:
    Weights: 26 GB
    KV cache: 2.3 GB (efficient)
    Total: 28.3 GB
    Batch size: ~52

Quantization + PagedAttention:
    Weights: 6.5 GB
    KV cache: 1.15 GB
    Total: 7.65 GB
    Batch size: ~260 (32× vs original!)
```

**Verdict:** Combine both for maximum efficiency!

### 2. PagedAttention vs Offloading (FlexGen)

**FlexGen:**
- Offload model weights and KV cache to CPU/disk
- Can handle sequences longer than GPU memory
- Very slow due to PCIe bandwidth

**Comparison (LLaMA-13B, 2048 tokens):**

| Metric | FlexGen | PagedAttention | Winner |
|--------|---------|----------------|---------|
| Max sequence length | Unlimited | GPU memory | ✅ FlexGen |
| Throughput | 0.2 req/s | 15 req/s | ✅ Paged |
| Latency | 10s | 0.6s | ✅ Paged |
| GPU memory | Minimal | Full | ❌ Paged |

**Use cases:**
- **FlexGen**: Extremely long sequences (16K+ tokens), low throughput OK
- **PagedAttention**: High throughput serving, sequences fit in GPU

**Can combine:** Use PagedAttention + offload oldest requests to CPU when GPU full

### 3. PagedAttention vs KV Cache Compression (H2O, StreamingLLM)

**H2O (Heavy-Hitter Oracle):**
- Evict less important KV cache entries
- Keep only "heavy hitters" (high attention scores)
- Reduces cache size but approximates attention

**StreamingLLM:**
- Keep initial tokens + recent window
- Discard middle tokens
- Enables infinite-length streaming

**Comparison:**

| Technique | Memory Saved | Quality | Exact Attention? |
|-----------|--------------|---------|------------------|
| H2O | 50-80% | Good | ❌ No (approx) |
| StreamingLLM | 90%+ | Good for streaming | ❌ No |
| PagedAttention | 85% (waste) | Perfect | ✅ Yes |

**Trade-off:**
- Compression: More memory savings, but approximate
- PagedAttention: Less savings, but exact

**Can combine:** Use PagedAttention + H2O for even more efficiency (with quality trade-off)

---

## Hybrid Approaches

### 1. PagedAttention + FlashAttention

**Best practice for production:**

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-13b",
    # PagedAttention settings
    block_size=16,
    max_num_batched_tokens=4096,
    # Enable FlashAttention (automatic in vLLM)
    use_flash_attn=True,
)

# Get 18× speedup (3× Flash + 6× Paged)
```

**Benefits:**
- ✅ Faster attention computation (FlashAttention)
- ✅ Better memory management (PagedAttention)
- ✅ Higher throughput (continuous batching)

### 2. PagedAttention + Quantization

**Maximum efficiency:**

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-13b",
    quantization="awq",  # INT4 weights
    kv_cache_dtype="fp8",  # FP8 KV cache
    block_size=16,
    max_num_batched_tokens=8192,
)

# Fit 10× more sequences in same GPU memory
```

**Use case:** Maximize batch size for highest throughput

### 3. PagedAttention + Tensor Parallelism

**For very large models:**

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-70b",
    tensor_parallel_size=4,  # Split across 4 GPUs
    block_size=16,
)

# Each GPU manages its own PagedAttention blocks
# KV cache is partitioned across GPUs
```

**Benefits:**
- ✅ Handle models larger than single GPU
- ✅ PagedAttention works per-GPU (independent)
- ✅ Linear scaling up to memory bandwidth limit

### 4. PagedAttention + Pipeline Parallelism

**For maximum throughput on large clusters:**

```python
# Not yet supported in vLLM, but conceptually:

# Pipeline stages share block pool
# Pass block IDs between stages (not actual KV data)
# Reduces communication overhead
```

### 5. Multi-Query Attention + PagedAttention

**For models with MQA/GQA:**

```python
# Models like Falcon, MPT already use MQA
llm = LLM(model="tiiuae/falcon-40b")  # MQA built-in

# Get combined benefits:
# - 40× less KV cache (from MQA)
# - 6× less waste (from PagedAttention)
# = 240× total memory efficiency!
```

---

## Decision Framework

### When to Use PagedAttention (vLLM)

✅ **Use PagedAttention if:**
- High-throughput serving (APIs, chatbots)
- Variable sequence lengths (some short, some long)
- Parallel sampling (N outputs per prompt)
- Beam search
- Shared prompts across many requests
- Need to maximize GPU utilization
- Production deployment with SLA requirements

### When PagedAttention May Not Help

❌ **Skip PagedAttention if:**
- Single request at a time (no batching benefit)
- All sequences are same length (less fragmentation)
- Very short sequences (<50 tokens) - overhead dominates
- Research/experimentation (simpler setup better)
- Extremely long sequences (>16K tokens) - consider offloading

### Choosing the Right System

```
┌─────────────────────────────────────────────────────────┐
│                  LLM Serving Decision Tree              │
└─────────────────────────────────────────────────────────┘

1. What's your use case?
   
   a) High-throughput API serving
      → Use vLLM (PagedAttention + continuous batching)
   
   b) Single-user chatbot
      → Use HuggingFace Transformers (simple)
   
   c) Research/experimentation
      → Use HuggingFace Transformers or TGI
   
   d) Extreme long context (>32K tokens)
      → Use FlexGen or LongLLM

2. Do you need parallel sampling?
   
   Yes → Use vLLM (memory sharing critical)
   No → Any system works

3. What's your model size?
   
   <7B → Single GPU, any system
   7-70B → Use vLLM with tensor parallelism
   >70B → Use vLLM with TP + pipeline parallelism

4. What's your batch size?
   
   1-4 → Any system works
   5-32 → Use continuous batching (Orca or vLLM)
   32+ → Use vLLM (PagedAttention essential)

5. What's your latency requirement?
   
   <100ms → Use vLLM + FlashAttention + quantization
   <1s → Use vLLM
   >1s → Any system works
```

### Performance Expectations

**Expected speedup over baseline (TGI):**

| Configuration | Throughput | Latency | Batch Size |
|---------------|------------|---------|------------|
| PagedAttention only | 6-8× | 1.5× | 6× |
| + FlashAttention | 18-24× | 2× | 6× |
| + Quantization | 30-40× | 2× | 10× |
| + Tensor Parallel | 50-80× | 1.5× | 8× |

### Cost Savings

**Example: Serving 1M requests/day**

Traditional (TGI):
- GPUs needed: 20× A100
- Cost: $20,000/month
- Latency: 2.5s avg

vLLM (PagedAttention):
- GPUs needed: 1× A100
- Cost: $1,000/month
- Latency: 0.6s avg

**Savings: $19,000/month (95%!)**

---

## Summary Table

| Technique | Type | Benefit | Compatible with Paged? | Quality Impact |
|-----------|------|---------|------------------------|----------------|
| **PagedAttention** | Memory mgmt | 6× less waste | - | None |
| **FlashAttention** | Computation | 3× faster | ✅ Yes | None |
| **Flash-Decoding** | Computation | 2× faster decode | ✅ Yes | None |
| **MQA/GQA** | Architecture | 40× less KV | ✅ Yes | Minimal |
| **Quantization** | Compression | 2-4× smaller | ✅ Yes | Minimal |
| **KV Quant** | Compression | 2× less KV | ✅ Yes | Minimal |
| **H2O** | Compression | 50-80% less KV | ✅ Yes | Some loss |
| **StreamingLLM** | Compression | 90% less KV | ✅ Yes | Some loss |
| **FlexGen** | Offloading | Infinite context | ❌ Different approach | None |
| **Continuous Batch** | Scheduling | 3× better util | ✅ Built-in vLLM | None |

---

## Key Takeaways

1. **PagedAttention addresses a different problem** than most optimizations:
   - Not about faster computation (that's FlashAttention)
   - Not about smaller models (that's quantization)
   - About eliminating memory waste in KV cache

2. **Best results from combining techniques:**
   - FlashAttention + PagedAttention = 18× speedup
   - Add quantization = 40× speedup
   - Add MQA = 240× memory efficiency

3. **vLLM is the production standard** for PagedAttention:
   - Proven in large-scale deployments
   - 20× speedup over baseline in real-world scenarios
   - Compatible with most popular models

4. **PagedAttention enables new capabilities:**
   - Efficient parallel sampling (shared memory)
   - Beam search without memory blowup
   - Continuous batching for high GPU utilization
   - Preemption and priority scheduling

5. **Choose based on your use case:**
   - High throughput → vLLM (PagedAttention)
   - Low latency → FlashAttention + vLLM
   - Maximum efficiency → Quantization + MQA + vLLM
   - Extreme context → FlexGen or custom solution

---

*For implementation details, see IMPLEMENTATION.md. For quick start, see QUICKSTART.md.*
