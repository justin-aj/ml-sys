"""
TASO Tutorial 1: Simple Algebraic Rewrite

Demonstrates the distributive property optimization:
    Before: Y = (A @ B) + (A @ C)
    After:  Y = A @ (B + C)

This reduces 2 matrix multiplications to 1, saving ~50% FLOPs and memory.
"""

import torch
import time
import numpy as np

# ============================================================
# Visualization
# ============================================================

def print_graph_visualization():
    """Show the computation graph transformation"""
    print("\n" + "="*70)
    print("📊 GRAPH TRANSFORMATION VISUALIZATION")
    print("="*70)
    
    print("\n📉 BEFORE: Y = (A @ B) + (A @ C)")
    print("""
    Computation Graph:
    
        A
       / \\
      @   @        ← Two separate matrix multiplications
     B     C
      \\   /
       +           ← Then add the results
       |
       Y
    
    Operations:
      1. X1 = matmul(A, B)    [M×K] @ [K×N] = [M×N]
      2. X2 = matmul(A, C)    [M×K] @ [K×N] = [M×N]
      3. Y = add(X1, X2)       [M×N] + [M×N] = [M×N]
    
    Cost:
      - FLOPs: 2×M×K×N (two matmuls) + M×N (one add)
      - Memory: 2×M×N (store both X1 and X2)
      - Kernel launches: 3
    """)
    
    print("\n⚡ AFTER: Y = A @ (B + C)  [TASO Optimized]")
    print("""
    Computation Graph:
    
        B   C
         \\ /
          +        ← First add the matrices (cheap!)
          |
          @        ← Then single matrix multiplication
          A
          |
          Y
    
    Operations:
      1. X1 = add(B, C)        [K×N] + [K×N] = [K×N]
      2. Y = matmul(A, X1)     [M×K] @ [K×N] = [M×N]
    
    Cost:
      - FLOPs: K×N (one add) + M×K×N (one matmul)
      - Memory: K×N (store only X1)
      - Kernel launches: 2
    """)
    
    print("\n💡 KEY INSIGHT:")
    print("    Distributive property: A@B + A@C = A@(B+C)")
    print("    Addition is MUCH cheaper than matrix multiplication!")
    print("    Moving the add before the matmul saves ~50% FLOPs\n")
    print("="*70 + "\n")


# ============================================================
# Original (Unoptimized) Implementation
# ============================================================

def compute_original(A, B, C):
    """
    Original computation: Y = (A @ B) + (A @ C)
    
    This is inefficient because it does TWO expensive matrix multiplications.
    """
    X1 = torch.matmul(A, B)  # First matmul
    X2 = torch.matmul(A, C)  # Second matmul
    Y = X1 + X2              # Add results
    return Y


# ============================================================
# TASO Optimized Implementation
# ============================================================

def compute_taso_optimized(A, B, C):
    """
    TASO optimized: Y = A @ (B + C)
    
    This is more efficient because it does ONE matrix multiplication.
    """
    X1 = B + C               # Add first (cheap!)
    Y = torch.matmul(A, X1)  # Then single matmul
    return Y


# ============================================================
# Correctness Verification
# ============================================================

def verify_correctness(M=128, K=256, N=128):
    """Verify that both implementations produce identical results"""
    print("\n" + "="*70)
    print("✅ CORRECTNESS VERIFICATION")
    print("="*70)
    
    # Create random test tensors
    torch.manual_seed(42)
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    C = torch.randn(K, N)
    
    print(f"\nTensor shapes:")
    print(f"  A: {list(A.shape)} = [{M}×{K}]")
    print(f"  B: {list(B.shape)} = [{K}×{N}]")
    print(f"  C: {list(C.shape)} = [{K}×{N}]")
    
    # Compute with both methods
    Y_original = compute_original(A, B, C)
    Y_optimized = compute_taso_optimized(A, B, C)
    
    # Check if results match
    max_diff = torch.max(torch.abs(Y_original - Y_optimized)).item()
    mean_diff = torch.mean(torch.abs(Y_original - Y_optimized)).item()
    
    print(f"\nResults:")
    print(f"  Original output shape: {list(Y_original.shape)}")
    print(f"  Optimized output shape: {list(Y_optimized.shape)}")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    
    if max_diff < 1e-5:
        print("\n  ✓ PASS: Results are identical (within numerical precision)")
    else:
        print("\n  ✗ FAIL: Results differ!")
    
    print("="*70 + "\n")
    return max_diff < 1e-5


# ============================================================
# Cost Analysis
# ============================================================

def analyze_costs(M, K, N):
    """Analyze FLOPs and memory costs"""
    print("\n" + "="*70)
    print("📊 COST ANALYSIS")
    print("="*70)
    
    print(f"\nProblem size: M={M}, K={K}, N={N}")
    
    # FLOPs calculation
    matmul_flops = 2 * M * K * N  # matmul is roughly 2MKN FLOPs
    add_flops = M * N
    small_add_flops = K * N
    
    original_flops = 2 * matmul_flops + add_flops
    optimized_flops = matmul_flops + small_add_flops
    
    print(f"\n📈 FLOPs (Floating Point Operations):")
    print(f"  Original:")
    print(f"    - MatMul(A,B): {matmul_flops:,} FLOPs")
    print(f"    - MatMul(A,C): {matmul_flops:,} FLOPs")
    print(f"    - Add(X1,X2):  {add_flops:,} FLOPs")
    print(f"    Total:         {original_flops:,} FLOPs")
    
    print(f"\n  TASO Optimized:")
    print(f"    - Add(B,C):    {small_add_flops:,} FLOPs")
    print(f"    - MatMul(A,X1):{matmul_flops:,} FLOPs")
    print(f"    Total:         {optimized_flops:,} FLOPs")
    
    flops_reduction = (1 - optimized_flops / original_flops) * 100
    print(f"\n  FLOPs Saved:   {original_flops - optimized_flops:,} FLOPs ({flops_reduction:.1f}%)")
    
    # Memory calculation
    element_size = 4  # FP32 = 4 bytes
    original_memory = 2 * M * N * element_size  # X1 and X2
    optimized_memory = K * N * element_size     # Only B+C
    
    print(f"\n💾 Intermediate Memory:")
    print(f"  Original:")
    print(f"    - X1 [M×N]:    {M*N:,} elements = {M*N*element_size/1024/1024:.2f} MB")
    print(f"    - X2 [M×N]:    {M*N:,} elements = {M*N*element_size/1024/1024:.2f} MB")
    print(f"    Total:         {original_memory/1024/1024:.2f} MB")
    
    print(f"\n  TASO Optimized:")
    print(f"    - X1 [K×N]:    {K*N:,} elements = {K*N*element_size/1024/1024:.2f} MB")
    print(f"    Total:         {optimized_memory/1024/1024:.2f} MB")
    
    memory_reduction = (1 - optimized_memory / original_memory) * 100
    print(f"\n  Memory Saved:  {(original_memory - optimized_memory)/1024/1024:.2f} MB ({memory_reduction:.1f}%)")
    
    print("\n🚀 Kernel Launches:")
    print(f"  Original:      3 kernels (matmul, matmul, add)")
    print(f"  TASO Optimized: 2 kernels (add, matmul)")
    print(f"  Reduction:     33% fewer kernel launches")
    
    print("="*70 + "\n")


# ============================================================
# Performance Benchmark
# ============================================================

def benchmark(M=1024, K=512, N=1024, warmup=10, iterations=100):
    """Benchmark both implementations"""
    print("\n" + "="*70)
    print("⏱️  PERFORMANCE BENCHMARK")
    print("="*70)
    
    # Create test tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device.upper()}")
    
    if device == 'cpu':
        print("⚠️  Warning: Running on CPU. GPU would show larger speedups.")
    
    torch.manual_seed(42)
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)
    C = torch.randn(K, N, device=device)
    
    print(f"Matrix sizes: A=[{M}×{K}], B=[{K}×{N}], C=[{K}×{N}]")
    
    # Warmup
    print(f"\nWarming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = compute_original(A, B, C)
        _ = compute_taso_optimized(A, B, C)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark original
    print(f"Benchmarking original ({iterations} iterations)...")
    start = time.perf_counter()
    for _ in range(iterations):
        Y = compute_original(A, B, C)
        if device == 'cuda':
            torch.cuda.synchronize()
    end = time.perf_counter()
    original_time = (end - start) / iterations * 1000  # ms
    
    # Benchmark optimized
    print(f"Benchmarking TASO optimized ({iterations} iterations)...")
    start = time.perf_counter()
    for _ in range(iterations):
        Y = compute_taso_optimized(A, B, C)
        if device == 'cuda':
            torch.cuda.synchronize()
    end = time.perf_counter()
    optimized_time = (end - start) / iterations * 1000  # ms
    
    # Results
    speedup = original_time / optimized_time
    
    print(f"\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    print(f"Original:        {original_time:.3f} ms")
    print(f"TASO Optimized:  {optimized_time:.3f} ms")
    print(f"Speedup:         {speedup:.2f}x")
    
    if speedup > 1.2:
        print(f"\n🎉 TASO optimization achieved {speedup:.2f}x speedup!")
    elif speedup > 1.0:
        print(f"\n✓ TASO optimization slightly faster ({speedup:.2f}x)")
    else:
        print(f"\n⚠️ No speedup on this hardware/size. Try larger matrices or GPU.")
        print("   (TASO wins bigger on GPUs and for larger problems)")
    
    print("="*70 + "\n")


# ============================================================
# Scaling Analysis
# ============================================================

def scaling_analysis():
    """Show how speedup scales with problem size"""
    print("\n" + "="*70)
    print("📈 SCALING ANALYSIS: Speedup vs Matrix Size")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]
    
    print(f"\nDevice: {device.upper()}")
    print(f"\nConfig: M=K=N (square matrices)\n")
    print("-"*70)
    print(f"{'Size':>8} | {'Original (ms)':>15} | {'TASO (ms)':>12} | {'Speedup':>8}")
    print("-"*70)
    
    for M, K, N in sizes:
        A = torch.randn(M, K, device=device)
        B = torch.randn(K, N, device=device)
        C = torch.randn(K, N, device=device)
        
        # Warmup
        for _ in range(5):
            _ = compute_original(A, B, C)
            _ = compute_taso_optimized(A, B, C)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Time original
        start = time.perf_counter()
        for _ in range(20):
            _ = compute_original(A, B, C)
            if device == 'cuda':
                torch.cuda.synchronize()
        original_time = (time.perf_counter() - start) / 20 * 1000
        
        # Time optimized
        start = time.perf_counter()
        for _ in range(20):
            _ = compute_taso_optimized(A, B, C)
            if device == 'cuda':
                torch.cuda.synchronize()
        optimized_time = (time.perf_counter() - start) / 20 * 1000
        
        speedup = original_time / optimized_time
        print(f"{M:>4}×{M:<3} | {original_time:>12.3f} ms | {optimized_time:>9.3f} ms | {speedup:>7.2f}x")
    
    print("-"*70)
    print("\n💡 Observation:")
    print("   Speedup generally increases with problem size!")
    print("   Larger matrices → matmul cost dominates → bigger TASO wins")
    print("="*70 + "\n")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎓 TASO TUTORIAL 1: DISTRIBUTIVE PROPERTY OPTIMIZATION")
    print("="*70)
    
    # Show the concept
    print_graph_visualization()
    
    # Verify correctness
    is_correct = verify_correctness(M=128, K=256, N=128)
    
    if not is_correct:
        print("❌ Correctness check failed! Stopping.")
        exit(1)
    
    # Analyze costs
    analyze_costs(M=1024, K=512, N=1024)
    
    # Benchmark
    benchmark(M=1024, K=512, N=1024, warmup=10, iterations=100)
    
    # Scaling analysis
    scaling_analysis()
    
    # Summary
    print("\n" + "="*70)
    print("✅ TUTORIAL COMPLETE!")
    print("="*70)
    print("""
Key Takeaways:
    
1. 🧮 Algebraic rewrites can eliminate operations
   - A@B + A@C → A@(B+C) saves one matmul
   
2. 💡 Addition is much cheaper than multiplication
   - Moving addition before matmul reduces FLOPs by ~50%
   
3. 💾 Memory savings are significant
   - Fewer intermediate tensors = less memory pressure
   
4. 🚀 Real speedups on GPUs
   - 1.5-2× faster for typical matrix sizes
   
5. 📈 Scales with problem size
   - Larger problems → bigger TASO wins!

Next Steps:
    - Try transformer_attention.py for real-world example
    - See EXAMPLES.md for more rewrite patterns
    - Read CONCEPT.md for deeper understanding
""")
    print("="*70 + "\n")
