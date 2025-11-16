# PagedAttention: Deep Dive into Concepts

## Table of Contents
- [Mathematical Foundations](#mathematical-foundations)
- [Block-based Memory Management](#block-based-memory-management)
- [Attention Computation with Blocks](#attention-computation-with-blocks)
- [Memory Sharing Mechanisms](#memory-sharing-mechanisms)
- [Scheduling and Batching](#scheduling-and-batching)
- [Performance Analysis](#performance-analysis)

---

## Mathematical Foundations

### Standard Self-Attention Recap

For a sequence of length $n$ with hidden dimension $d$:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:
- $Q \in \mathbb{R}^{n \times d_k}$ (Queries)
- $K \in \mathbb{R}^{n \times d_k}$ (Keys)
- $V \in \mathbb{R}^{n \times d_v}$ (Values)
- $d_k$ is the key dimension (typically $d_k = d / h$ where $h$ is number of heads)

### KV Cache in Autoregressive Generation

During token-by-token generation at step $t$:

**Without KV cache** (inefficient):
$$
\text{Attention}_t(q_t, [k_1, ..., k_t], [v_1, ..., v_t])
$$
- Must recompute $k_i, v_i$ for all $i \in [1, t]$
- Complexity: $O(t \cdot d)$ per step → $O(n^2 \cdot d)$ total

**With KV cache** (efficient):
$$
\text{Attention}_t(q_t, K_{\text{cache}}, V_{\text{cache}})
$$

Where:
$$
K_{\text{cache}} = [k_1, k_2, ..., k_t] \quad \text{(concatenated over time)}
$$
$$
V_{\text{cache}} = [v_1, v_2, ..., v_t]
$$

- Compute only $k_t, v_t$ at step $t$
- Complexity: $O(d)$ per step → $O(n \cdot d)$ total

**Memory cost per sequence**:
$$
M_{\text{KV}} = 2 \times L \times n \times h \times d_k \times p
$$

Where:
- $L$ = number of layers
- $n$ = sequence length
- $h$ = number of attention heads
- $d_k$ = key/value dimension per head
- $p$ = precision (2 bytes for FP16, 4 bytes for FP32)

**Example (LLaMA-13B)**:
$$
M_{\text{KV}} = 2 \times 40 \times 2048 \times 40 \times 128 \times 2 = 1,677,721,600 \text{ bytes} \approx 1.6 \text{ GB}
$$

---

## Block-based Memory Management

### Block Structure

Define block size $B$ (typically $B = 16$). Each physical block stores:

$$
\text{Block}_{\text{physical}} = 
\begin{bmatrix}
k_0 & k_1 & \cdots & k_{B-1} \\
v_0 & v_1 & \cdots & v_{B-1}
\end{bmatrix}
\in \mathbb{R}^{2 \times B \times h \times d_k}
$$

For $L$ layers, each block actually stores:
$$
\text{Block}_{\text{physical}} \in \mathbb{R}^{L \times 2 \times B \times h \times d_k}
$$

**Memory per block**:
$$
M_{\text{block}} = L \times 2 \times B \times h \times d_k \times p
$$

For LLaMA-13B with $B=16$:
$$
M_{\text{block}} = 40 \times 2 \times 16 \times 40 \times 128 \times 2 = 13,107,200 \text{ bytes} \approx 12.5 \text{ MB}
$$

### Logical vs Physical Blocks

**Critical distinction**: The block table maps **fixed-size token ranges** to physical blocks.

For a sequence of length $n$ with block size $B$:

**Number of blocks needed**:
$$
N_{\text{blocks}} = \lceil n / B \rceil
$$

**Block table** (simple array):
$$
\phi = [\text{PhysBlock}_0, \text{PhysBlock}_1, \ldots, \text{PhysBlock}_{N-1}]
$$

**Mapping is always fixed-size**:
- Block table index 0 → Tokens $[0, B-1]$ → Physical block $\phi[0]$
- Block table index 1 → Tokens $[B, 2B-1]$ → Physical block $\phi[1]$
- Block table index 2 → Tokens $[2B, 3B-1]$ → Physical block $\phi[2]$
- ...

**Example** (sequence length $n = 50$, block size $B = 16$):
$$
N_{\text{blocks}} = \lceil 50/16 \rceil = 4
$$
$$
\phi = [7, 2, 11, 15] \quad \text{(physical block IDs)}
$$

**Token-to-block lookup**:
- Tokens 0-15 stored in Physical Block 7
- Tokens 16-31 stored in Physical Block 2
- Tokens 32-47 stored in Physical Block 11
- Tokens 48-49 stored in Physical Block 15 (partially filled)

**The block table never stores individual token indices**—it only stores which physical block contains each fixed-size chunk.

### Memory Allocation

**Traditional contiguous allocation**:
$$
M_{\text{allocated}} = M_{\text{KV}}(n_{\text{max}}) = 2 \times L \times n_{\text{max}} \times h \times d_k \times p
$$

**PagedAttention block allocation**:
$$
M_{\text{allocated}} = \lceil n_{\text{actual}} / B \rceil \times M_{\text{block}}
$$

**Waste comparison**:

Traditional:
$$
\text{Waste}_{\text{traditional}} = M_{\text{KV}}(n_{\text{max}}) - M_{\text{KV}}(n_{\text{actual}})
$$
$$
= 2 \times L \times (n_{\text{max}} - n_{\text{actual}}) \times h \times d_k \times p
$$

PagedAttention:
$$
\text{Waste}_{\text{paged}} = (B - (n_{\text{actual}} \mod B)) \times \frac{M_{\text{block}}}{B}
$$
$$
= (B - (n_{\text{actual}} \mod B)) \times 2 \times L \times h \times d_k \times p
$$

**Waste ratio**:
$$
\frac{\text{Waste}_{\text{paged}}}{\text{Waste}_{\text{traditional}}} = \frac{B - (n_{\text{actual}} \mod B)}{n_{\text{max}} - n_{\text{actual}}}
$$

For $n_{\text{actual}} \ll n_{\text{max}}$:
$$
\approx \frac{B}{n_{\text{max}} - n_{\text{actual}}} \approx \frac{16}{2048 - 300} \approx 0.009 \quad \text{(0.9\%)}
$$

---

## Attention Computation with Blocks

### Block-wise Attention Algorithm

Given query $q_t$ at step $t$ and block table $\phi$:

$$
\text{Attention}(q_t) = \sum_{i=0}^{N_{\text{logical}}-1} \text{AttentionBlock}(q_t, \text{Block}_{\phi(i)})
$$

More precisely:

**Step 1: Compute scores per block**
$$
s_{i,j} = \frac{q_t \cdot k_{\phi(i),j}}{\sqrt{d_k}} \quad \text{for } j \in [0, B-1]
$$

Where $k_{\phi(i),j}$ is the $j$-th key in physical block $\phi(i)$.

**Step 2: Global softmax**
$$
\alpha_{i,j} = \frac{\exp(s_{i,j})}{\sum_{i'=0}^{N_{\text{logical}}-1} \sum_{j'=0}^{B-1} \exp(s_{i',j'})}
$$

**Step 3: Weighted sum**
$$
o = \sum_{i=0}^{N_{\text{logical}}-1} \sum_{j=0}^{B-1} \alpha_{i,j} \cdot v_{\phi(i),j}
$$

### Computational Complexity

**Per-token complexity**:
- Score computation: $O(n \cdot d_k)$
- Softmax: $O(n)$
- Weighted sum: $O(n \cdot d_v)$
- **Total**: $O(n \cdot d)$ (same as traditional!)

**Memory access pattern**:
- Traditional: Sequential access to contiguous KV cache
- PagedAttention: Gather from multiple physical blocks

**Key insight**: Block-based organization adds minimal computational overhead (~3-5%) due to:
1. Block table lookups are fast (simple array indexing)
2. Modern GPUs handle non-contiguous memory efficiently
3. CUDA kernel optimization (fused operations)

---

## Memory Sharing Mechanisms

### Reference Counting

Each physical block has a reference counter:
$$
\text{RefCount}(b) = \text{number of sequences using block } b
$$

**Allocation rule**:
$$
\text{RefCount}(b) := \text{RefCount}(b) + 1 \quad \text{when block } b \text{ is assigned}
$$

**Deallocation rule**:
$$
\text{Free}(b) \iff \text{RefCount}(b) = 0
$$

### Prefix Sharing

For sequences sharing a common prefix of length $p$:

**Logical blocks shared**: $N_{\text{shared}} = \lfloor p / B \rfloor$

**Memory saved per additional sequence**:
$$
\Delta M = N_{\text{shared}} \times M_{\text{block}}
$$

Example (10 sequences, prefix length 256, $B=16$):
$$
N_{\text{shared}} = \lfloor 256 / 16 \rfloor = 16 \text{ blocks}
$$
$$
\text{Traditional memory} = 10 \times 16 \times 12.5 \text{ MB} = 2000 \text{ MB}
$$
$$
\text{PagedAttention memory} = 16 \times 12.5 \text{ MB} = 200 \text{ MB}
$$
$$
\text{Savings} = 1800 \text{ MB} \quad (90\%)
$$

### Copy-on-Write (CoW)

When a shared block needs modification:

**CoW operation**:
1. Check: $\text{RefCount}(b) > 1$?
2. If yes:
   - Allocate new block $b'$
   - Copy: $\text{Block}_{b'} \leftarrow \text{Block}_b$
   - Update block table: $\phi(i) \leftarrow b'$
   - Decrement: $\text{RefCount}(b) := \text{RefCount}(b) - 1$
   - Initialize: $\text{RefCount}(b') := 1$
3. Modify $b'$ (not $b$)

**Cost analysis**:
- Copy cost: $O(M_{\text{block}})$ (12.5 MB for LLaMA-13B)
- Frequency: Once per divergence point
- Amortized: Negligible compared to generation time

### Prefix Tree (Trie) for Sharing

Sequences can be organized in a **radix tree** based on prefixes:

```
                    [Root]
                      |
                [System Prompt] (shared)
                   /    \
              [User1]  [User2]
               /  \      /  \
           [Ans1][Ans2][Ans3][Ans4]
```

**Sharing ratio**:
$$
\text{Sharing}(\text{node}) = \frac{\text{RefCount}(\text{node})}{\text{Total sequences}}
$$

**Average memory per sequence**:
$$
M_{\text{avg}} = \sum_{\text{depth } d} \frac{M_{\text{blocks}}(d)}{\text{RefCount}(d)}
$$

---

## Scheduling and Batching

### Continuous Batching

Traditional static batching:
$$
\text{Batch} = \{s_1, s_2, ..., s_B\} \quad \text{(fixed until all complete)}
$$

PagedAttention enables **continuous batching**:
$$
\text{Batch}_t = \text{Batch}_{t-1} \cup \text{New}_t \setminus \text{Finished}_t
$$

**Throughput gain**:
$$
\text{Throughput}_{\text{continuous}} = \frac{\text{GPU time utilized}}{\text{Total time}} \times \text{Batch size}
$$

With static batching:
- GPU idle when waiting for slowest sequence
- Utilization: ~60-70%

With continuous batching:
- GPU always processing full batch
- Utilization: ~90-95%

**Effective throughput**: $1.5 \times$ improvement from batching alone

### Preemption and Swapping

When memory is full, PagedAttention can **preempt** sequences:

**Preemption algorithm**:
1. Select victim sequence $s_{\text{victim}}$ (e.g., lowest priority)
2. Swap blocks to CPU/disk:
   $$
   \text{Blocks}_{\text{CPU}} \leftarrow \text{Blocks}_{\text{GPU}}(\phi_s)
   $$
3. Free GPU blocks:
   $$
   \text{RefCount}(b) := \text{RefCount}(b) - 1 \quad \forall b \in \phi_s
   $$
4. Resume later by swapping back

**Cost**:
- Swap time: $O(M_{\text{KV}})$
- PCIe bandwidth: ~12-16 GB/s
- Swap time for 1.6 GB: ~100-130 ms

**Benefit**: Handle bursty traffic without OOM errors

---

## Performance Analysis

### Memory Efficiency

**Utilization metric**:
$$
U = \frac{\text{Actual tokens stored}}{\text{Total token slots allocated}}
$$

**Traditional**:
$$
U_{\text{traditional}} = \frac{\sum_{i=1}^N n_i}{N \times n_{\text{max}}} = \frac{\bar{n}}{n_{\text{max}}}
$$

For $\bar{n} = 300$, $n_{\text{max}} = 2048$:
$$
U_{\text{traditional}} = \frac{300}{2048} \approx 0.146 \quad (14.6\%)
$$

**PagedAttention**:
$$
U_{\text{paged}} = \frac{\sum_{i=1}^N n_i}{\sum_{i=1}^N (\lceil n_i / B \rceil \times B)} \approx \frac{n}{n + B/2}
$$

For $n = 300$, $B = 16$:
$$
U_{\text{paged}} \approx \frac{300}{300 + 8} \approx 0.974 \quad (97.4\%)
$$

**Improvement**:
$$
\frac{U_{\text{paged}}}{U_{\text{traditional}}} = \frac{0.974}{0.146} \approx 6.7 \times
$$

### Throughput Model

**Throughput** (requests per second):
$$
T = \frac{B_{\text{eff}} \times \text{GPU time fraction}}{t_{\text{gen}}}
$$

Where:
- $B_{\text{eff}}$ = effective batch size
- $t_{\text{gen}}$ = time to generate one token

**Batch size scaling**:
$$
B_{\text{eff}} = \frac{M_{\text{GPU}} - M_{\text{model}}}{M_{\text{KV per seq}}}
$$

Traditional:
$$
B_{\text{eff, trad}} = \frac{M_{\text{GPU}} - M_{\text{model}}}{M_{\text{KV}}(n_{\text{max}})}
$$

PagedAttention:
$$
B_{\text{eff, paged}} = \frac{M_{\text{GPU}} - M_{\text{model}}}{M_{\text{KV}}(\bar{n}) \times 1.03}
$$

For A100 (80 GB), LLaMA-13B (26 GB weights), $n_{\text{max}}=2048$, $\bar{n}=300$:
$$
B_{\text{eff, trad}} = \frac{80 - 26}{1.6} \approx 34
$$
$$
B_{\text{eff, paged}} = \frac{80 - 26}{0.24 \times 1.03} \approx 218
$$

**Throughput gain**:
$$
\frac{T_{\text{paged}}}{T_{\text{trad}}} = \frac{218}{34} \approx 6.4 \times
$$

### Latency Analysis

**Time to first token (TTFT)**:
$$
\text{TTFT} = t_{\text{prefill}} = \frac{n_{\text{prompt}} \times d}{R_{\text{compute}}}
$$

PagedAttention **does not improve** TTFT (same computation).

**Time per output token (TPOT)**:
$$
\text{TPOT} = t_{\text{decode}} = \frac{d}{R_{\text{compute}}} + \frac{M_{\text{KV read}}}{R_{\text{memory}}}
$$

PagedAttention adds **minimal overhead** (~3-5%) from:
- Block table lookups: $O(N_{\text{logical}})$
- Non-contiguous memory access

**Overall latency** (for output length $m$):
$$
\text{Latency} = \text{TTFT} + m \times \text{TPOT}
$$

With batching, per-request latency increases, but **throughput** increases significantly.

---

## Key Mathematical Insights

### 1. Memory Waste Reduction

$$
\text{Waste reduction} = \frac{n_{\text{max}} - n_{\text{actual}}}{B - (n_{\text{actual}} \mod B)} \approx \frac{n_{\text{max}}}{B}
$$

For typical values ($n_{\text{max}} = 2048$, $B = 16$):
$$
\approx 128 \times \text{ reduction}
$$

### 2. Prefix Sharing Efficiency

For $N$ sequences sharing prefix of length $p$:
$$
\text{Memory}_{\text{shared}} = p \times M_{\text{token}} \times (1 + \frac{N-1}{N}) \approx p \times M_{\text{token}}
$$
$$
\text{Memory}_{\text{unshared}} = p \times M_{\text{token}} \times N
$$
$$
\text{Savings} = (1 - \frac{1}{N}) \times 100\%
$$

For $N=10$: **90% savings** on shared prefix.

### 3. Block Size Trade-off

Smaller $B$:
- ✅ Less internal fragmentation
- ❌ More blocks → larger block tables
- ❌ More memory access overhead

Larger $B$:
- ✅ Fewer blocks → smaller tables
- ✅ Better memory locality
- ❌ More internal fragmentation

**Optimal** $B \approx 16$ (empirically determined):
- Fragmentation: $\frac{B}{2} / n \approx \frac{8}{300} \approx 2.7\%$
- Block table size: Negligible
- Memory access: Efficient

### 4. Computational Overhead

Block-based attention overhead:
$$
\text{Overhead} = \frac{t_{\text{block lookup}} + t_{\text{gather}}}{t_{\text{attention}}}
$$

Empirically:
$$
\text{Overhead} \approx 3-5\%
$$

With optimized CUDA kernels (fused operations), overhead can be reduced to <2%.

---

## Production-Scale Memory Management

### The Million-User Problem

**Challenge**: Serve millions of concurrent users when GPU memory cannot hold all KV caches.

#### Realistic Numbers for 100B Model

| Parameter | Value |
|-----------|-------|
| Model size | 100B parameters |
| Hidden dimension $d$ | 12,288 |
| Layers $L$ | 96 |
| Max sequence length $n_{\text{max}}$ | 2,048 |
| Precision $p$ | 2 bytes (FP16) |

**KV cache per sequence**:
$$
M_{\text{seq}} = 2 \times L \times n_{\text{max}} \times d \times p
$$
$$
= 2 \times 96 \times 2048 \times 12288 \times 2
$$
$$
\approx 9.66 \times 10^{10} \text{ bytes} \approx 90 \text{ GB}
$$

For full 2048-token sequences, each user needs ~90 GB KV cache!

### Three-Tier Memory Hierarchy

Production systems use **hierarchical storage**:

```
┌─────────────────────────────────────────────────────┐
│  Tier 1: GPU Memory (Hot - Active Generation)       │
│  - Capacity: 80 GB per H100                         │
│  - Latency: <1 ms                                   │
│  - Active users: ~1,000 per GPU                     │
├─────────────────────────────────────────────────────┤
│  Tier 2: CPU RAM (Warm - Recent Users)              │
│  - Capacity: 512 GB - 2 TB per server               │
│  - Latency: 10-50 ms (PCIe transfer)                │
│  - Recent users: ~10,000 per server                 │
├─────────────────────────────────────────────────────┤
│  Tier 3: NVMe SSD (Cold - Inactive Sessions)        │
│  - Capacity: 10+ TB per server                      │
│  - Latency: 100-500 ms                              │
│  - Total users: Millions                            │
└─────────────────────────────────────────────────────┘
```

### Mathematical Model

**Setup**: Cluster with $G$ GPUs, each with memory $M_{\text{GPU}}$

**Active users in GPU**:
$$
U_{\text{active}} = \frac{G \times M_{\text{GPU}} \times \alpha}{M_{\text{seq,avg}}}
$$

Where:
- $\alpha$ = fraction of GPU memory allocated to KV cache (typically 0.6-0.7)
- $M_{\text{seq,avg}}$ = average KV cache size per sequence

**Example** (1000 H100 GPUs, 80 GB each):
$$
M_{\text{GPU}} = 80 \text{ GB}, \quad G = 1000, \quad \alpha = 0.7
$$

For sequences with average 512 tokens (not max 2048):
$$
M_{\text{seq,avg}} = \frac{90 \text{ GB} \times 512}{2048} = 22.5 \text{ GB}
$$

Active users:
$$
U_{\text{active}} = \frac{1000 \times 80 \times 0.7}{22.5} \approx 2,489 \text{ users}
$$

**Only ~2,500 users** can have KV cache in GPU simultaneously!

### Eviction and Swapping Strategy

**Least Recently Used (LRU) with aging**:

1. **Track last access time** $t_{\text{access}}$ for each sequence
2. **Priority score**:
   $$
   \text{Priority}(s) = \frac{1}{t_{\text{current}} - t_{\text{access}}(s)}
   $$
3. **Eviction**: When GPU memory needed, evict lowest priority sequence

**Block-level eviction**:
- Don't evict entire sequence at once
- Evict older blocks first (tokens 0-15, then 16-31, etc.)
- Keep recent blocks in GPU for faster resume

**Eviction cost**:
$$
T_{\text{evict}} = \frac{M_{\text{seq}}}{B_{\text{PCIe}}}
$$

For 90 GB KV cache, PCIe 4.0 bandwidth ~16 GB/s:
$$
T_{\text{evict}} = \frac{90}{16} \approx 5.6 \text{ seconds}
$$

But with block-level eviction (only evict 10 GB):
$$
T_{\text{evict}} = \frac{10}{16} \approx 0.6 \text{ seconds}
$$

### Request Handling Flow

**Cold start** (user KV cache in NVMe):
```
1. Request arrives
2. Check block table: KV pages in NVMe
3. Fetch required blocks: NVMe → CPU RAM (200ms)
4. Fetch to GPU: CPU RAM → GPU (50ms)
5. Compute new token: GPU (5ms)
6. Total latency: ~255ms
```

**Warm start** (user KV cache in CPU RAM):
```
1. Request arrives  
2. Check block table: KV pages in CPU RAM
3. Fetch to GPU: CPU RAM → GPU (50ms)
4. Compute new token: GPU (5ms)
5. Total latency: ~55ms
```

**Hot** (user KV cache in GPU):
```
1. Request arrives
2. Check block table: KV pages in GPU
3. Compute new token: GPU (5ms)
4. Total latency: ~5ms
```

### Serving 1 Million Users

**Assumptions**:
- 1,000 H100 GPUs
- Average sequence: 512 tokens
- User activity: 1% active at any moment

**Capacity breakdown**:
$$
\text{GPU active}: 2,500 \text{ users (hot)}
$$
$$
\text{CPU RAM active}: 25,000 \text{ users (warm, last 5 min)}
$$
$$
\text{NVMe storage}: 972,500 \text{ users (cold)}
$$
$$
\text{Total}: 1,000,000 \text{ users}
$$

**Latency distribution**:
- 0.25% requests: ~5ms (already in GPU)
- 2.5% requests: ~55ms (in CPU RAM)
- 97.25% requests: ~255ms (cold start from NVMe)

**Why this works**:
- Only 0.25% of users generate tokens simultaneously
- PagedAttention's block-based design enables:
  - Partial eviction (old blocks first)
  - Fast swapping (move blocks, not full sequences)
  - Instant reuse (freed blocks immediately available)

### Key Insight

**Without PagedAttention**:
- Must allocate contiguous space
- Cannot partially evict sequences
- Fragmentation prevents efficient reuse
- **Result**: Need 10-100× more GPUs for same user count

**With PagedAttention**:
- Block-based allocation enables partial eviction
- Block table tracks across GPU/CPU/NVMe seamlessly
- Zero fragmentation = instant reuse
- **Result**: Serve millions with thousands of GPUs

---

## Summary

PagedAttention achieves:

1. **$128\times$ waste reduction** through block-based allocation
2. **90%+ memory savings** for shared prefixes
3. **6-8× batch size increase** enabling higher throughput
4. **<5% computational overhead** with optimized kernels
5. **Flexible scheduling** with preemption and swapping

The key is applying **virtual memory principles** to KV cache management, turning a memory bottleneck into an efficient, scalable system.

---

*Next: See IMPLEMENTATION.md for code examples and COMPARISON.md for detailed benchmarks.*
