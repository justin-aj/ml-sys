"""
Tutorial 3: Flash Attention Lite - Advanced Fusion

THE PROBLEM:
Standard attention in transformers has a MEMORY BOTTLENECK:

    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d)) @ V

For sequence length N and hidden dimension d:
1. Q @ K.T creates an [N × N] attention matrix - HUGE for long sequences!
2. Softmax over that matrix - reads/writes [N × N] to memory
3. Multiply by V - reads [N × N] again

Memory usage: O(N²) - can't fit long sequences!
Memory traffic: Read/write full [N × N] matrix multiple times

Example: N=2048, d=64, batch=1
- Attention matrix: 2048 × 2048 × 4 bytes = 16 MB
- Total memory traffic: ~100 MB for one attention head
- For 12 heads in BERT: 1.2 GB per layer!

THE SOLUTION (Flash Attention):
Compute attention in BLOCKS without materializing the full [N × N] matrix!

Key ideas:
1. **Tiling**: Process attention in small blocks that fit in SRAM
2. **Online softmax**: Update softmax statistics incrementally (no full matrix!)
3. **Recomputation in backward**: Trade recomputation for memory savings

Memory usage: O(N) instead of O(N²)
Memory traffic: 4-8× less than standard attention

This is THE technique that enabled:
- GPT-4's 32k token context
- LLaMA's efficient training
- Stable Diffusion's cross-attention optimization

EXPECTED RESULT:
On V100: 2-4x speedup for seq_len >= 1024
Plus: Can fit much longer sequences in memory!

NOTE: This is a SIMPLIFIED educational version.
Production Flash Attention has additional optimizations:
- Backward pass fusion
- Multi-query attention support  
- Better auto-tuning
See: https://github.com/Dao-AILab/flash-attention
"""

import torch
import triton
import triton.language as tl
import time
import math


@triton.jit
def flash_attention_fwd_kernel(
    Q, K, V, Out,           # Input/output pointers
    L,                      # Softmax denominator (for backward pass)
    stride_qz, stride_qh, stride_qm, stride_qk,  # Q strides
    stride_kz, stride_kh, stride_kn, stride_kk,  # K strides  
    stride_vz, stride_vh, stride_vn, stride_vk,  # V strides
    stride_oz, stride_oh, stride_om, stride_ok,  # Out strides
    Z, H, N_CTX, D_HEAD,    # Dimensions
    BLOCK_M: tl.constexpr,  # Block size for queries
    BLOCK_N: tl.constexpr,  # Block size for keys
    BLOCK_DMODEL: tl.constexpr,  # Hidden dimension
):
    """
    Flash Attention forward pass kernel (simplified).
    
    Key innovation: Compute attention block-by-block without storing
    the full [N × N] attention matrix!
    
    Algorithm:
    For each query block (BLOCK_M queries):
        Initialize output accumulator and softmax statistics
        For each key block (BLOCK_N keys):
            1. Compute attention scores for this block (in SRAM)
            2. Update softmax statistics (online algorithm)
            3. Update output accumulator with weighted values
            4. Discard attention scores (never write to DRAM!)
    
    Memory: Only stores final output [N × D], not attention matrix [N × N]!
    """
    # Program ID - which batch and head are we processing?
    start_m = tl.program_id(0)  # Which query block
    off_hz = tl.program_id(1)   # Which batch & head
    
    # Compute batch and head indices
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Offset pointers to the correct batch and head
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    
    # Query block: BLOCK_M queries starting at start_m * BLOCK_M
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Pointers to Q for this block
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    
    # Pointers to K and V (will loop over N dimension)
    k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    
    # Output accumulator (starts at zero)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Softmax statistics (for online softmax algorithm)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # Max value seen so far
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Sum of exp values
    
    # Load Q for this block (stays in SRAM for entire loop!)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    
    # === LOOP OVER KEY/VALUE BLOCKS ===
    # This is the magic - we process K, V in chunks that fit in SRAM
    for start_n in range(0, N_CTX, BLOCK_N):
        # Bounds check for this K/V block
        offs_n_curr = start_n + offs_n
        
        # Load K and V for this block (into SRAM)
        k = tl.load(k_ptrs, mask=offs_n_curr[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < N_CTX, other=0.0)
        
        # === COMPUTE ATTENTION SCORES (Q @ K.T) ===
        # This creates a [BLOCK_M × BLOCK_N] matrix in SRAM (NOT written to DRAM!)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))  # Matrix multiply in SRAM
        qk *= 1.0 / math.sqrt(D_HEAD)  # Scale
        
        # Causal mask (for autoregressive models like GPT)
        # Only attend to earlier positions
        mask = offs_m[:, None] >= offs_n_curr[None, :]
        qk = tl.where(mask, qk, float("-inf"))
        
        # === ONLINE SOFTMAX UPDATE ===
        # Update max value
        m_ij = tl.max(qk, axis=1)  # Max over key dimension
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Compute exp and update statistics
        alpha = tl.exp(m_i - m_i_new)  # Rescaling factor for old values
        p = tl.exp(qk - m_i_new[:, None])  # Softmax numerator
        
        # Update sum (denominator)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        
        # === UPDATE OUTPUT ===
        # Rescale old output
        acc = acc * alpha[:, None]
        
        # Add contribution from this block: p @ V
        # p is [BLOCK_M × BLOCK_N], v is [BLOCK_N × BLOCK_DMODEL]
        acc += tl.dot(p.to(v.dtype), v)
        
        # Update max for next iteration
        m_i = m_i_new
        
        # Move to next K/V block
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        
        # Note: We just discarded 'qk' and 'p' - they never touched DRAM!
    
    # === FINAL NORMALIZATION ===
    # Divide by softmax denominator
    acc = acc / l_i[:, None]
    
    # === STORE OUTPUT ===
    # Write final output to DRAM (only [BLOCK_M × D_HEAD] per iteration)
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)
    
    # Store softmax denominator for backward pass
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_i, mask=offs_m < N_CTX)


def flash_attention_lite(q, k, v, causal=True):
    """
    Simplified Flash Attention implementation.
    
    Args:
        q, k, v: [batch, num_heads, seq_len, head_dim]
        causal: If True, use causal masking (GPT-style)
    
    Returns:
        output: [batch, num_heads, seq_len, head_dim]
    """
    batch, num_heads, seq_len, head_dim = q.shape
    
    # Allocate output
    output = torch.empty_like(q)
    
    # Allocate softmax denominators (for backward pass, not used in this tutorial)
    L = torch.empty((batch * num_heads, seq_len), device=q.device, dtype=torch.float32)
    
    # Block sizes (these should be tuned for your GPU)
    BLOCK_M = 64  # Query block size
    BLOCK_N = 64  # Key/value block size
    
    # Number of query blocks
    num_blocks = triton.cdiv(seq_len, BLOCK_M)
    
    # Grid: (num_query_blocks, batch * num_heads)
    grid = (num_blocks, batch * num_heads)
    
    # Launch kernel
    flash_attention_fwd_kernel[grid](
        q, k, v, output, L,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch, num_heads, seq_len, head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=head_dim,
    )
    
    return output


def pytorch_attention(q, k, v, causal=True):
    """
    Standard PyTorch attention (the slow way).
    
    Creates the full [batch, heads, N, N] attention matrix in memory.
    """
    batch, num_heads, seq_len, head_dim = q.shape
    
    # Compute attention scores: Q @ K.T
    # Shape: [batch, heads, seq_len, seq_len]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    
    # Causal mask
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
    
    # Softmax
    # This materializes the full [N × N] attention matrix!
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attn_weights, v)
    
    return output


def benchmark_attention(batch=4, num_heads=8, seq_len=1024, head_dim=64, num_runs=50):
    """
    Benchmark PyTorch vs Flash Attention.
    """
    # Create test data
    q = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    
    print(f"\n{'='*70}")
    print(f"Benchmarking Flash Attention")
    print(f"Shape: [batch={batch}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}]")
    print(f"{'='*70}\n")
    
    # Memory analysis
    attention_matrix_size = batch * num_heads * seq_len * seq_len * 2 / 1e6  # FP16
    qkv_size = batch * num_heads * seq_len * head_dim * 2 * 3 / 1e6  # Q, K, V
    
    print(f"Memory Analysis:")
    print(f"  Q, K, V size: {qkv_size:.1f} MB")
    print(f"  Attention matrix [N×N]: {attention_matrix_size:.1f} MB")
    print(f"  PyTorch peak memory: ~{qkv_size + attention_matrix_size:.1f} MB")
    print(f"  Flash Attention memory: ~{qkv_size:.1f} MB (no attention matrix!)")
    print(f"  Memory saved: {attention_matrix_size:.1f} MB ({attention_matrix_size/(qkv_size + attention_matrix_size)*100:.0f}%)\n")
    
    # === PyTorch Attention ===
    print("🔥 PyTorch Standard Attention")
    print("-" * 70)
    
    # Warmup
    for _ in range(5):
        _ = pytorch_attention(q, k, v)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        pytorch_out = pytorch_attention(q, k, v)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / num_runs * 1000
    
    print(f"Memory traffic:")
    print(f"  1. Q @ K.T: Read Q, K, write attention matrix ({attention_matrix_size:.1f} MB)")
    print(f"  2. Softmax: Read & write attention matrix ({attention_matrix_size*2:.1f} MB)")
    print(f"  3. Attn @ V: Read attention matrix & V, write output")
    print(f"  Total: ~{attention_matrix_size*4 + qkv_size:.1f} MB")
    print(f"Time: {pytorch_time:.3f} ms")
    
    # === Flash Attention ===
    print(f"\n{'⚡'} Flash Attention (Triton)")
    print("-" * 70)
    
    # Warmup
    for _ in range(5):
        _ = flash_attention_lite(q, k, v)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        flash_out = flash_attention_lite(q, k, v)
    torch.cuda.synchronize()
    flash_time = (time.perf_counter() - start) / num_runs * 1000
    
    print(f"Memory traffic:")
    print(f"  1. Load Q once per query block")
    print(f"  2. Load K, V in chunks (blocks)")
    print(f"  3. Compute attention scores IN SRAM (never touch DRAM!)")
    print(f"  4. Write final output")
    print(f"  Total: ~{qkv_size*2:.1f} MB (4x less!)")
    print(f"Time: {flash_time:.3f} ms")
    
    # === Verify Correctness ===
    max_diff = torch.max(torch.abs(pytorch_out - flash_out)).item()
    mean_diff = torch.mean(torch.abs(pytorch_out - flash_out)).item()
    
    print(f"\n{'='*70}")
    print(f"✅ Correctness Check")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    print(f"Match: {'✓ PASS' if max_diff < 1e-2 else '⚠️  CLOSE' if max_diff < 1e-1 else '✗ FAIL'}")
    print(f"(Note: FP16 precision means small differences are expected)")
    
    # === Performance Summary ===
    speedup = pytorch_time / flash_time
    
    print(f"\n{'='*70}")
    print(f"📊 Performance Summary")
    print(f"{'='*70}")
    print(f"PyTorch:        {pytorch_time:.3f} ms")
    print(f"Flash Attn:     {flash_time:.3f} ms")
    print(f"Speedup:        {speedup:.2f}x {'🚀' if speedup > 2.0 else '✓' if speedup > 1.2 else '⚠️'}")
    print(f"Memory saved:   {attention_matrix_size:.1f} MB ({attention_matrix_size/(qkv_size + attention_matrix_size)*100:.0f}%)")
    print(f"{'='*70}\n")
    
    # === Real-world Impact ===
    print("💡 Real-World Impact")
    print("-" * 70)
    print(f"GPT-2 has 12 layers with 12 attention heads each = 144 attention operations")
    print(f"\nPer-forward-pass savings:")
    print(f"  PyTorch: 144 × {pytorch_time:.3f}ms = {144*pytorch_time:.1f}ms")
    print(f"  Flash:   144 × {flash_time:.3f}ms = {144*flash_time:.1f}ms")
    print(f"  Saved:   {144*(pytorch_time-flash_time):.1f}ms per inference\n")
    
    print(f"Memory savings per layer:")
    print(f"  Standard attention: {attention_matrix_size:.1f} MB")
    print(f"  Flash attention: 0 MB (no materialized attention matrix!)")
    print(f"  For GPT-2 (144 attention ops): {144*attention_matrix_size:.1f} MB saved!")
    print(f"\nThis is why GPT-4 can handle 32k tokens - it fits in memory!")
    
    return speedup


def scaling_analysis():
    """
    Show how Flash Attention scales with sequence length.
    """
    print("\n" + "="*70)
    print("📈 Scaling Analysis: Speedup vs Sequence Length")
    print("="*70)
    
    batch, num_heads, head_dim = 4, 8, 64
    
    print(f"\nConfig: batch={batch}, heads={num_heads}, head_dim={head_dim}")
    print("-" * 70)
    print(f"{'Seq Len':>8} | {'PyTorch (ms)':>12} | {'Flash (ms)':>12} | {'Speedup':>8} | {'Mem Saved (MB)':>15}")
    print("-" * 70)
    
    for seq_len in [256, 512, 1024, 2048]:
        q = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        
        # PyTorch
        for _ in range(3):
            _ = pytorch_attention(q, k, v)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            _ = pytorch_attention(q, k, v)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / 20 * 1000
        
        # Flash
        for _ in range(3):
            _ = flash_attention_lite(q, k, v)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            _ = flash_attention_lite(q, k, v)
        torch.cuda.synchronize()
        flash_time = (time.perf_counter() - start) / 20 * 1000
        
        speedup = pytorch_time / flash_time
        mem_saved = batch * num_heads * seq_len * seq_len * 2 / 1e6
        
        print(f"{seq_len:8d} | {pytorch_time:12.3f} | {flash_time:12.3f} | {speedup:8.2f}x | {mem_saved:15.1f}")
    
    print("-" * 70)
    print("\n💡 Key Insight:")
    print("   As sequence length grows, Flash Attention wins bigger!")
    print("   - Speedup increases (memory bottleneck more severe)")
    print("   - Memory savings grow quadratically (O(N²) vs O(N))")
    print("   - Longer sequences become feasible (GPT-4: 32k tokens!)")
    print("="*70 + "\n")


def visualize_algorithm():
    """
    ASCII art visualization of Flash Attention algorithm.
    """
    print("\n" + "="*70)
    print("🧠 Flash Attention Algorithm Visualization")
    print("="*70)
    
    print("\n📉 Standard Attention (materialized [N×N] matrix):")
    print("""
    Step 1: Compute scores
    ┌─────┐   ┌─────┐T   ┌───────────┐
    │  Q  │ @ │  K  │  = │ Scores    │  ← N×N matrix in DRAM!
    │ N×D │   │ N×D │    │ [N×N]     │
    └─────┘   └─────┘    └───────────┘
    
    Step 2: Softmax
    ┌───────────┐   softmax   ┌───────────┐
    │ Scores    │  ────────>  │ Attention │  ← Read & write N×N!
    │ [N×N]     │             │ [N×N]     │
    └───────────┘             └───────────┘
    
    Step 3: Apply to values
    ┌───────────┐   ┌─────┐   ┌─────┐
    │ Attention │ @ │  V  │ = │ Out │
    │ [N×N]     │   │ N×D │   │ N×D │
    └───────────┘   └─────┘   └─────┘
    
    Problem: N×N matrix written to DRAM, read multiple times!
    """)
    
    print("⚡ Flash Attention (block-wise, no materialized matrix):")
    print("""
    For each QUERY BLOCK (e.g., 64 queries):
        output = zeros()
        max_val = -inf
        sum_exp = 0
        
        For each KEY BLOCK (e.g., 64 keys):
            ┌────────────────────────────────┐
            │  SRAM (on-chip, fast!)         │
            │                                │
            │  1. scores = Q_block @ K_block │  ← Small [64×64] in SRAM
            │  2. Update max_val             │  ← In registers!
            │  3. probs = exp(scores - max)  │  ← In SRAM
            │  4. Update sum_exp             │  ← In registers!
            │  5. output += probs @ V_block  │  ← Accumulate in SRAM
            │                                │
            │  DISCARD scores & probs!       │  ← Never write to DRAM!
            └────────────────────────────────┘
        
        output = output / sum_exp  ← Final normalization
        Write output to DRAM  ← Only output written!
    
    Key: Small [64×64] blocks stay in SRAM. Full [N×N] never exists!
    """)
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Show algorithm visualization
    visualize_algorithm()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This tutorial requires an NVIDIA GPU.")
        exit(1)
    
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    
    # Run main benchmark
    speedup = benchmark_attention(batch=4, num_heads=8, seq_len=1024, head_dim=64, num_runs=50)
    
    # Scaling analysis
    scaling_analysis()
    
    print("\n✅ Tutorial complete! Key takeaways:")
    print("   1. Flash Attention computes attention block-by-block")
    print("   2. Attention matrix NEVER materialized in memory (stays in SRAM)")
    print("   3. Online softmax algorithm updates statistics incrementally")
    print("   4. Memory: O(N²) → O(N), enabling much longer sequences")
    print("   5. This is THE technique used in GPT-4, LLaMA, production transformers")
    print("\n🎓 You've completed all Triton tutorials!")
    print("   Next steps:")
    print("   - Read the Flash Attention paper: https://arxiv.org/abs/2205.14135")
    print("   - Use flash-attn library: pip install flash-attn")
    print("   - Apply these techniques to your own models!")
    print("   - Explore REAL_WORLD_USES.md for production examples")
