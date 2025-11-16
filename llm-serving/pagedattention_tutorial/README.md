# PagedAttention Tutorial

## 🎯 Quick Start: Run on V100 GPU!

**Two ways to see PagedAttention in action:**

### Option 1: Memory Calculator (Recommended Start)

```bash
cd llm-serving/pagedattention_tutorial
pip install torch matplotlib
python paged_attention_benchmark.py  # 30 seconds
```

**What it does:** Calculates memory requirements mathematically
- ✅ Fast (30 seconds)
- ✅ No vLLM needed
- ✅ Educational (shows the math)
- ✅ Works on CPU or GPU

**You'll see:**
- 77% memory waste eliminated with variable-length requests
- 4x more concurrent users on V100 (32GB)
- 83% memory saved with prefix sharing

### Option 2: Real vLLM Benchmark (Validation)

```bash
pip install vllm  # Requires CUDA 11.8+, Linux/WSL
python vllm_real_benchmark.py --test all  # 2-5 minutes
```

**What it does:** Actually runs vLLM inference
- ✅ Real throughput measurements
- ✅ Actual GPU memory usage
- ✅ Proves PagedAttention works
- ⚠️ Requires vLLM installed

**You'll see:**
- Actual requests/second and latency
- Real GPU memory utilization
- Batch scaling efficiency

### Comparison

| Feature | Memory Calculator | Real vLLM Benchmark |
|---------|-------------------|---------------------|
| **Runtime** | 30 seconds | 2-5 minutes |
| **Requirements** | torch, matplotlib | vllm (CUDA, Linux) |
| **What it measures** | Memory calculations | Actual performance |
| **Best for** | Learning concept | Validation |
| **Works on CPU?** | Yes (reduced scale) | No (GPU only) |

**Start with Option 1** to understand the concept, then use **Option 2** if you want real numbers!

---

See [QUICKSTART.md](QUICKSTART.md) for details and [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) to record your V100 results!

---

## Table of Contents
- [Introduction](#introduction)
- [The KV Cache Problem](#the-kv-cache-problem)
- [PagedAttention Solution](#pagedattention-solution)
- [Core Concepts](#core-concepts)
- [Implementation Details](#implementation-details)
- [Performance Benefits](#performance-benefits)
- [Real-World Impact](#real-world-impact)
- [Comparison with Other Methods](#comparison-with-other-methods)

## Introduction

**PagedAttention** is a revolutionary attention algorithm developed by UC Berkeley for efficient LLM inference serving. It's the core innovation behind **vLLM** (very Large Language Model), which achieves **2-24x higher throughput** than traditional serving systems like HuggingFace Text Generation Inference (TGI) and FasterTransformer.

### What Problem Does It Solve?

When serving LLMs (like GPT, LLaMA, etc.), the **KV cache** (key-value cache) for attention becomes the memory bottleneck:
- **Training**: Uses ~90% memory for model weights, ~10% for activations
- **Serving**: Uses ~70% memory for KV cache, ~30% for model weights

PagedAttention solves the **memory fragmentation** and **over-allocation** problems in KV cache management.

---

## The KV Cache Problem

### What is KV Cache?

In transformer models, during inference, we cache the attention keys (K) and values (V) from previous tokens to avoid recomputation:

```
Input tokens:  [t1, t2, t3, ..., tn]
                 ↓   ↓   ↓       ↓
Cached KV:     [K1, K2, K3, ..., Kn]
               [V1, V2, V3, ..., Vn]
```

For each new token generation:
- **Without KV cache**: Recompute attention for ALL previous tokens (O(n²) every step)
- **With KV cache**: Only compute attention for the new token (O(n) every step)

### Memory Requirements

For a single sequence in a 13B parameter model (like LLaMA-13B):

```
KV cache per token = 2 (K+V) × num_layers × hidden_size × precision
                   = 2 × 40 × 5120 × 2 bytes (FP16)
                   = 819,200 bytes ≈ 0.8 MB per token

For 2048 tokens:
                   = 0.8 MB × 2048 = 1.6 GB per sequence
```

### Traditional Problems

#### 1. **Internal Fragmentation**

Traditional systems pre-allocate a fixed-size buffer for the maximum sequence length:

```
Allocated: [████████████████████████████████████████] 2048 tokens
Used:      [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 256 tokens (12.5%)
Wasted:    [░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓] 1792 tokens (87.5%)
```

**Problem**: We don't know the output length in advance, so we must allocate for worst-case (max_seq_len), wasting memory.

#### 2. **External Fragmentation**

When sequences finish, they leave gaps that can't be reused efficiently:

```
GPU Memory:
Seq1: [████░░░░]  finished
Seq2: [██████████] active
Seq3: [████░░░░░░] finished
Seq4: [██████████] active
      ▲▲▲▲        ▲▲▲▲▲
      gaps can't fit new sequence
```

#### 3. **Over-Reservation**

Real-world example from the vLLM paper:
- **Allocated**: 2048 tokens per sequence
- **Actually used**: Average ~15% (varies widely: 4%-100%)
- **Wasted**: ~85% of allocated KV cache memory

---

## PagedAttention Solution

### Core Idea: Virtual Memory Paging

PagedAttention borrows the concept of **virtual memory** from operating systems:

```
Operating System         PagedAttention
─────────────           ───────────────
Virtual Memory    →     Logical token sequence
Physical Memory   →     Physical KV blocks
Page Table        →     Block Table
```

### Key Innovation: Fixed-Size Blocks

Instead of contiguous memory allocation, **split KV cache into fixed-size blocks** (e.g., 16 tokens per block):

```
Traditional (contiguous):
Seq1: [K0...K2047, V0...V2047] ← Must pre-allocate for max_seq_len

PagedAttention (block-based):
Seq1: Block0[K0...K15, V0...V15]   → Physical block 7
      Block1[K16...K31, V16...V31]  → Physical block 2
      Block2[K32...K47, V32...V47]  → Physical block 11
      ...
      Allocate blocks on-demand as tokens generate!
```

**Critical**: Blocks are **always fixed-size** (e.g., 16 tokens). This is non-negotiable—it's what eliminates fragmentation.

### Block Table: The Index

Each sequence has a **block table** (a simple array/list) that maps logical token ranges to physical block IDs:

```
Sequence 1's Block Table: [7, 2, 11, 15]
  Tokens 0-15   → Physical Block 7
  Tokens 16-31  → Physical Block 2
  Tokens 32-47  → Physical Block 11
  Tokens 48-63  → Physical Block 15

Sequence 2's Block Table: [3, 9]
  Tokens 0-15   → Physical Block 3
  Tokens 16-31  → Physical Block 9
```

**The block table never tracks individual tokens**—it only tracks which blocks store which fixed ranges of tokens.

---

## Core Concepts

### 1. Block Size

Typical block size: **16 tokens** (fixed)

```
Block = [K0...K15, V0...V15]  ← Exactly 16 tokens, always

Memory per block (13B model):
= 16 tokens × 0.8 MB/token
= 12.8 MB per block
```

**Why fixed-size matters**: 
- Eliminates fragmentation (any free block can serve any sequence)
- Enables instant reuse (no copying or consolidation needed)
- Simplifies memory management (GPU memory = pool of identical blocks)

### 2. Block Allocation

Blocks are allocated **on-demand, one token at a time**:

```
Sequence starts with prompt "Hello world":
  Token 0: Generate → Allocate Block 0, store K0, V0
  Token 1: Generate → Store K1, V1 in Block 0
  ...
  Token 15: Generate → Store K15, V15 in Block 0 (now full)
  Token 16: Generate → Allocate Block 1, store K16, V16
  ...
```

**Key insight**: Inference generates **one token at a time**. Tokens accumulate in the current block until it's full, then a new block is allocated. No batching, no arbitrary slices like "tokens 1-3 in block A."

### 3. How Attention Reads Blocks

During attention computation for a new token:

```
1. Follow block table: [7, 2, 11] for this sequence
2. Read Block 7 → get K0...K15, V0...V15
3. Read Block 2 → get K16...K31, V16...V31  
4. Read Block 11 → get K32...K47, V32...V47
5. Compute attention: new_token attends to all 48 previous tokens
```

The attention kernel **follows the block table** to fetch KV data in logical token order, even though physically the blocks are scattered in GPU memory.

### 3. Block Sharing (for Prompt)

Multiple sequences can **share** the same physical blocks for common prompts:

```
Prompt: "Translate to French: "

Seq1: "Translate to French: Hello"
Seq2: "Translate to French: Goodbye"
      ─────────────────────
      Shared prefix blocks!

Block Table Seq1: [5, 12, 8]   ← Block 5 shared
                   ↑
Block Table Seq2: [5, 12, 3]   ← Block 5 shared
                   ↑
```

This is crucial for:
- **Parallel sampling**: Generate multiple outputs for one prompt
- **Beam search**: Explore multiple generation paths
- **Shared system prompts**: Same prefix for many requests

### 4. Copy-on-Write

When a shared block needs to be modified, use **copy-on-write**:

```
Before (shared):
Seq1 → Block 5
Seq2 → Block 5

After generation (Seq1 adds token):
Seq1 → Block 5 (copy) → Block 17 (new)
Seq2 → Block 5 (original)
```

### 5. Memory Pool Management

PagedAttention maintains a **free block pool**:

```
Free blocks: [1, 4, 6, 10, 13, 14, 16, ...]

Request arrives → Pop from pool
Sequence ends  → Push to pool

No fragmentation! All blocks are same size.
```

---

## Implementation Details

### Attention Computation

Traditional attention with KV cache:

```python
# Traditional: KV cache is contiguous
def attention(Q, K_cache, V_cache):
    # K_cache: [batch, seq_len, num_heads, head_dim]
    # V_cache: [batch, seq_len, num_heads, head_dim]
    
    scores = Q @ K_cache.transpose(-2, -1)  # [batch, 1, seq_len]
    scores = scores / sqrt(head_dim)
    attn_weights = softmax(scores)
    output = attn_weights @ V_cache          # [batch, 1, head_dim]
    return output
```

PagedAttention with blocks:

```python
# PagedAttention: KV cache is paged
def paged_attention(Q, block_tables, kv_cache_blocks):
    """
    Q: [num_seqs, num_heads, head_dim] - queries for new token
    block_tables: [num_seqs, max_num_blocks] - block mapping
    kv_cache_blocks: [num_blocks, block_size, num_heads, head_dim, 2]
                     - all physical blocks (K and V)
    """
    outputs = []
    
    for seq_idx in range(num_seqs):
        # Get blocks for this sequence
        block_ids = block_tables[seq_idx]
        
        # Gather K, V from blocks
        K_seq = []
        V_seq = []
        for block_id in block_ids:
            if block_id >= 0:  # Valid block
                K_seq.append(kv_cache_blocks[block_id, :, :, :, 0])
                V_seq.append(kv_cache_blocks[block_id, :, :, :, 1])
        
        K_seq = concat(K_seq, dim=0)  # [seq_len, num_heads, head_dim]
        V_seq = concat(V_seq, dim=0)
        
        # Standard attention
        scores = Q[seq_idx] @ K_seq.transpose(-2, -1)
        scores = scores / sqrt(head_dim)
        attn_weights = softmax(scores)
        output = attn_weights @ V_seq
        outputs.append(output)
    
    return stack(outputs)
```

### Optimized CUDA Kernel

The actual vLLM implementation uses a custom **CUDA kernel** that:
1. **Fuses** block lookup and attention computation
2. **Optimizes** memory access patterns
3. **Minimizes** data movement between blocks

```cuda
// Pseudocode for optimized kernel
__global__ void paged_attention_kernel(
    const float* Q,              // Query
    const float* K_blocks,       // Physical K blocks
    const float* V_blocks,       // Physical V blocks
    const int* block_table,      // Block mapping
    float* output
) {
    int seq_idx = blockIdx.x;
    int head_idx = threadIdx.x;
    
    // Load query for this sequence and head
    float q[HEAD_DIM];
    load_query(Q, seq_idx, head_idx, q);
    
    // Iterate through blocks
    float scores[MAX_SEQ_LEN];
    int token_idx = 0;
    
    for (int block_idx = 0; block_table[seq_idx][block_idx] >= 0; block_idx++) {
        int physical_block_id = block_table[seq_idx][block_idx];
        
        // Compute attention scores for all tokens in this block
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float k[HEAD_DIM];
            load_key(K_blocks, physical_block_id, i, head_idx, k);
            scores[token_idx++] = dot_product(q, k);
        }
    }
    
    // Softmax
    softmax(scores, token_idx);
    
    // Compute weighted sum of values
    float output_vec[HEAD_DIM] = {0};
    token_idx = 0;
    
    for (int block_idx = 0; block_table[seq_idx][block_idx] >= 0; block_idx++) {
        int physical_block_id = block_table[seq_idx][block_idx];
        
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float v[HEAD_DIM];
            load_value(V_blocks, physical_block_id, i, head_idx, v);
            
            for (int d = 0; d < HEAD_DIM; d++) {
                output_vec[d] += scores[token_idx] * v[d];
            }
            token_idx++;
        }
    }
    
    // Store output
    store_output(output, seq_idx, head_idx, output_vec);
}
```

---

## Performance Benefits

### 1. Memory Efficiency

**Without PagedAttention** (traditional):
```
Batch size = 8 sequences
Max length = 2048 tokens
Average actual length = 300 tokens

Memory allocated:
= 8 × 2048 × 0.8 MB = 13.1 GB

Memory used:
= 8 × 300 × 0.8 MB = 1.9 GB

Waste: 11.2 GB (85%!)
```

**With PagedAttention**:
```
Block size = 16 tokens
Average length = 300 tokens

Blocks needed per sequence = ceil(300/16) = 19 blocks
Total blocks = 8 × 19 = 152 blocks

Memory used:
= 152 × 16 × 0.8 MB = 1.95 GB

Waste: Only last block partially filled (~0.05 GB, 2.5%)
```

### 2. Throughput Improvement

From the vLLM paper (LLaMA-13B on A100):

| System | Throughput (req/s) | Speedup |
|--------|-------------------|---------|
| HuggingFace TGI | 0.8 | 1x |
| FasterTransformer | 1.2 | 1.5x |
| Orca | 2.5 | 3.1x |
| **vLLM (PagedAttention)** | **19.3** | **24x** |

### 3. Batch Size Scaling

With better memory utilization, vLLM can handle **larger batch sizes**:

```
Traditional system:
Max batch size = 8 sequences (GPU OOM at batch size 9)

vLLM with PagedAttention:
Max batch size = 64+ sequences (5-8x larger batches)

Throughput ∝ Batch size (with good scheduling)
```

### 4. Memory Sharing Benefits

For **parallel sampling** (generating N outputs for one prompt):

```
Traditional:
N outputs = N × full KV cache for prompt

PagedAttention:
N outputs = 1 × KV cache for prompt (shared)
          + N × KV cache for unique suffixes

Savings = (N-1) × prompt_length × memory_per_token
```

Example (N=5, prompt=500 tokens, output=100 tokens each):
- **Traditional**: 5 × 600 tokens = 3000 token-memories
- **PagedAttention**: 1 × 500 + 5 × 100 = 1000 token-memories
- **Savings**: 67%!

---

## Real-World Impact

### Production Serving: The Million-User Problem

**Question**: How can a system serve **millions of concurrent users** when GPU memory can't hold all their KV caches?

**Answer**: Aggressive memory management with paging, eviction, and offloading.

#### The Raw Numbers (100B Parameter Model)

Let's use GPT-style 100B parameter model as example:

| Parameter | Value |
|-----------|-------|
| Model size | 100B parameters |
| Hidden dimension | 12,288 |
| Number of layers | 96 |
| Max sequence length | 2,048 tokens |
| GPU | H100 (80 GB) |
| KV cache per token per layer | ~1.2 MB |

**KV cache per sequence**:
```
KV_seq = 1.2 MB/token/layer × 2,048 tokens × 96 layers
       ≈ 236 GB per sequence
```

**This is larger than a single H100!** You **cannot** naively store complete KV caches in GPU memory.

#### What Production Systems Actually Do

**Setup**: 1,000 H100 GPUs = 80 TB total GPU memory

**Strategy**:

1. **Paged KV Cache**
   - Split KV cache into small blocks (512-1024 tokens per block)
   - Only **recent/active blocks** kept in GPU
   - Older blocks offloaded to CPU RAM or NVMe SSD

2. **Evict Inactive Users**
   - User hasn't generated tokens in last X seconds?
   - Move their KV cache blocks to **CPU RAM or NVMe**
   - Free GPU memory for **active users**

3. **Memory Juggling Example**
   ```
   Scenario: Shorter sequences (avg 1,024 tokens)
   
   Active KV cache per user ≈ 20 GB (with optimizations)
   
   80 TB GPU memory ÷ 20 GB/user = 4,000 users actively generating
   
   Remaining users: KV caches in CPU RAM/NVMe
   ```

4. **Request Handling Flow**
   ```
   New request arrives:
   1. Check if user's KV pages are in GPU
   2. If not → Fetch from CPU/NVMe (adds ~10ms latency)
   3. Compute new token
   4. Evict old/inactive KV pages to make room
   5. Store new KV in GPU blocks
   ```

5. **The Million-User Math**
   ```
   Total users: 1,000,000
   Active at once (in GPU): 4,000
   Inactive (in CPU/NVMe): 996,000
   
   Active percentage: 0.4%
   
   Result: Millions can interact, but only a tiny fraction
           have KV cache in GPU memory at any moment.
   ```

#### Why This Works

- **GPU**: Fast but expensive (~4,000 active users)
- **CPU RAM**: Slower but larger (~100,000 recently active users)
- **NVMe SSD**: Slowest but huge (~millions of total users)

**PagedAttention enables this** because:
- ✅ Fixed-size blocks can be moved/evicted individually
- ✅ Block tables make it trivial to track where data lives
- ✅ No fragmentation means instant reuse of freed blocks
- ✅ Partial sequences can be evicted (don't need full cache in GPU)

### Real Production Systems

**vLLM** implements this full strategy:

```python
# vLLM configuration for production scale
llm = LLM(
    model="gpt-100b",
    tensor_parallel_size=8,      # Split across 8 GPUs
    max_num_seqs=512,            # Max active sequences in GPU
    block_size=16,               # 16 tokens per block
    swap_space=100,              # GB of CPU RAM for swapping
    gpu_memory_utilization=0.9,  # Use 90% of GPU for KV cache
)
```

**Chatbot Arena (LMSYS)** serves millions of requests:
- Only active conversations have GPU KV cache
- Inactive chats swapped to CPU within seconds
- Can resume any conversation by fetching KV pages from CPU/disk

**Anyscale** production deployment:
- 1,000+ GPUs serving 10M+ daily requests
- <1% of user sessions have KV cache in GPU at any moment
- Median latency: ~100ms (including occasional page fetch)

### Key Takeaway

**You cannot store all KV caches in GPU memory at scale.**

Production systems use a **3-tier memory hierarchy**:
1. **GPU** (hot, active users) - milliseconds
2. **CPU RAM** (warm, recent users) - 10-50ms
3. **NVMe/Disk** (cold, inactive users) - 100-500ms

PagedAttention's block-based design makes this **memory juggling efficient**:
- Move individual blocks, not entire sequences
- No copying or consolidation overhead
- Instant reuse of freed GPU blocks
- Block tables update in nanoseconds

**Without PagedAttention**: You'd need 200× more GPUs or serve 200× fewer users.

### vLLM Serving System

PagedAttention is implemented in **vLLM**, which provides:

1. **High Throughput**: 2-24x faster than FasterTransformer/TGI
2. **Memory Efficiency**: Near-zero waste (~3% vs ~85%)
3. **Flexible Scheduling**: Continuous batching, preemption, swapping
4. **Easy Integration**: Compatible with HuggingFace models

### Use Cases

#### 1. **High-Traffic API Serving**
```
Scenario: Serve GPT-style chatbot to millions of users

Traditional:
- 8 concurrent requests per GPU
- High latency spikes

vLLM:
- 64+ concurrent requests per GPU
- Smooth latency, 8x lower cost
```

#### 2. **Parallel Sampling**
```
Scenario: Generate 10 different responses for each prompt

Traditional:
- Batch size 8 → Only 0.8 prompts processed concurrently
- Memory waste on duplicated prompt KV cache

vLLM:
- Batch size 80 (8 prompts × 10 samples)
- Share prompt KV cache across samples
- 10x higher throughput
```

#### 3. **Beam Search**
```
Scenario: Beam width = 5 for better quality

Traditional:
- Must replicate KV cache 5 times
- Memory bottleneck

vLLM:
- Share common prefix blocks
- Copy-on-write for diverged paths
- Minimal memory overhead
```

### Production Deployments

vLLM is used by:
- **Chatbot Arena** (LMSYS) - Serving millions of requests
- **Anyscale** - Ray-based LLM serving platform
- **Modal Labs** - Serverless LLM inference
- **Together.ai** - Decentralized inference network

---

## Comparison with Other Methods

### 1. PagedAttention vs. Traditional (Contiguous) KV Cache

| Aspect | Traditional | PagedAttention |
|--------|------------|----------------|
| **Allocation** | Pre-allocate max_seq_len | On-demand blocks |
| **Memory waste** | ~85% (internal fragmentation) | ~3% (last block only) |
| **Sharing** | Not possible | Yes (copy-on-write) |
| **Fragmentation** | Internal + External | Minimal |
| **Complexity** | Simple | Moderate (block table) |

### 2. PagedAttention vs. FlashAttention

**FlashAttention**: Optimizes attention **computation** (tiling, IO-awareness)
**PagedAttention**: Optimizes KV cache **memory management**

They are **complementary**! vLLM uses **FlashAttention + PagedAttention**:
- FlashAttention: Faster attention kernel
- PagedAttention: Better memory utilization

Combined speedup: **FlashAttention (2-4x) × PagedAttention (2-24x) = 4-96x!**

### 3. PagedAttention vs. Other Memory Optimizations

#### Quantization (e.g., GPTQ, AWQ)
- **Target**: Model weights (reduce from FP16 to INT4/INT8)
- **Benefit**: 2-4x weight memory savings
- **Compatibility**: Can combine with PagedAttention

#### Offloading (e.g., FlexGen)
- **Target**: Move KV cache to CPU/disk when not needed
- **Benefit**: Handle longer sequences than GPU memory
- **Trade-off**: Slow (requires data movement)
- **PagedAttention advantage**: Keeps everything in GPU, no swapping overhead

#### Compression (e.g., H2O, StreamingLLM)
- **Target**: Compress or evict KV cache entries
- **Benefit**: Reduce KV cache size
- **Trade-off**: May hurt quality (approximation)
- **PagedAttention advantage**: No quality loss, exact attention

### 4. PagedAttention vs. Continuous Batching

**Continuous Batching** (Orca): Dynamic batching as requests arrive/complete

PagedAttention **enables** better continuous batching:
```
Traditional:
- Fixed batch, can't add new requests until batch completes
- Memory fragmentation makes it hard to fit new requests

PagedAttention:
- Add/remove requests dynamically
- No fragmentation, easy to fit new requests
```

vLLM combines both: **Continuous Batching + PagedAttention**

---

## Key Takeaways

### 1. **Core Innovation**
PagedAttention applies **virtual memory paging** to KV cache management, eliminating fragmentation and enabling memory sharing.

### 2. **Main Benefits**
- **Memory efficiency**: ~85% → ~3% waste
- **Higher throughput**: 2-24x speedup
- **Larger batches**: 5-8x more concurrent requests
- **Memory sharing**: Efficient parallel sampling and beam search

### 3. **How It Works**
- Split KV cache into **fixed-size blocks** (e.g., 16 tokens)
- Use **block tables** to map logical to physical blocks
- Allocate blocks **on-demand** as sequences grow
- **Share blocks** for common prefixes (with copy-on-write)
- **Pool management** prevents fragmentation

### 4. **When to Use**
- ✅ High-throughput LLM serving (APIs, chatbots)
- ✅ Parallel sampling (multiple outputs per prompt)
- ✅ Beam search
- ✅ Long context windows with variable lengths
- ✅ Shared prompts across requests

### 5. **Complementary Technologies**
- Combine with **FlashAttention** for computation speedup
- Compatible with **quantization** for model weight reduction
- Enhanced by **continuous batching** for dynamic workloads

### 6. **Production Ready**
vLLM with PagedAttention is production-ready and widely deployed, powering some of the largest LLM serving systems.

---

## Further Reading

- **Original Paper**: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)
- **vLLM GitHub**: https://github.com/vllm-project/vllm
- **vLLM Documentation**: https://docs.vllm.ai/
- **Blog Post**: "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention"

## Next Steps

1. Read **CONCEPTS.md** for detailed mathematical analysis
2. Explore **IMPLEMENTATION.md** for code examples
3. Check **COMPARISON.md** for in-depth comparisons
4. See **QUICKSTART.md** to run vLLM with PagedAttention

---

*This tutorial is part of the Distributed Training & Inference series. For training parallelism, see ZeRO, Megatron-LM, and PipeDream tutorials.*
