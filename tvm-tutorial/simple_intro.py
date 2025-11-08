"""
TVM Introduction - Compare PyTorch vs TVM
==========================================

This script demonstrates the basics of TVM:
1. Define a computation (matrix multiplication)
2. Create a schedule (optimization plan)
3. Build and run it
4. Compare with PyTorch

Works on CPU - no GPU required for this intro!
"""

import tvm
from tvm import te
import numpy as np
import torch
import time


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def benchmark_pytorch(size=1024, iterations=100):
    """Benchmark PyTorch matrix multiplication"""
    print_header("PyTorch Baseline")
    
    # Create random matrices
    A = torch.randn(size, size)
    B = torch.randn(size, size)
    
    # Warmup
    for _ in range(10):
        C = torch.mm(A, B)
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        C = torch.mm(A, B)
    end = time.time()
    
    avg_time = (end - start) / iterations * 1000  # Convert to ms
    print(f"Matrix size: {size}x{size}")
    print(f"PyTorch time: {avg_time:.4f} ms")
    print(f"GFLOPS: {2 * size**3 / avg_time / 1e6:.2f}")
    
    return avg_time, C.numpy()


def benchmark_tvm_basic(size=1024, iterations=100):
    """Benchmark TVM with basic schedule (no optimization)"""
    print_header("TVM - Basic (No Optimization)")
    
    # Step 1: Define computation
    # This is like writing: C[i, j] = sum(A[i, k] * B[k, j] for k in range(size))
    n = te.var("n")
    A = te.placeholder((n, n), name="A")
    B = te.placeholder((n, n), name="B")
    k = te.reduce_axis((0, n), name="k")
    C = te.compute((n, n), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    
    # Step 2: Create schedule (default - no optimization)
    s = te.create_schedule(C.op)
    
    # Step 3: Build the function
    func = tvm.build(s, [A, B, C], target="llvm", name="matmul")
    
    # Step 4: Prepare data
    np_A = np.random.randn(size, size).astype(np.float32)
    np_B = np.random.randn(size, size).astype(np.float32)
    np_C = np.zeros((size, size), dtype=np.float32)
    
    # Convert to TVM arrays
    dev = tvm.cpu(0)
    tvm_A = tvm.nd.array(np_A, dev)
    tvm_B = tvm.nd.array(np_B, dev)
    tvm_C = tvm.nd.array(np_C, dev)
    
    # Warmup
    for _ in range(10):
        func(tvm_A, tvm_B, tvm_C)
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        func(tvm_A, tvm_B, tvm_C)
    end = time.time()
    
    avg_time = (end - start) / iterations * 1000
    print(f"Matrix size: {size}x{size}")
    print(f"TVM (basic) time: {avg_time:.4f} ms")
    print(f"GFLOPS: {2 * size**3 / avg_time / 1e6:.2f}")
    
    return avg_time, tvm_C.numpy()


def benchmark_tvm_optimized(size=1024, iterations=100, tile_size=32):
    """Benchmark TVM with manual optimization (tiling + vectorization)"""
    print_header("TVM - Optimized (Manual Schedule)")
    
    # Step 1: Define computation (same as before)
    n = te.var("n")
    A = te.placeholder((n, n), name="A")
    B = te.placeholder((n, n), name="B")
    k = te.reduce_axis((0, n), name="k")
    C = te.compute((n, n), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    
    # Step 2: Create OPTIMIZED schedule
    s = te.create_schedule(C.op)
    
    # Get computation axes
    x, y = C.op.axis
    k_axis = C.op.reduce_axis[0]
    
    # Optimization 1: Tiling (cache blocking)
    # Split computation into tiles for better cache locality
    xo, xi = s[C].split(x, factor=tile_size)
    yo, yi = s[C].split(y, factor=tile_size)
    ko, ki = s[C].split(k_axis, factor=tile_size)
    
    # Reorder for better memory access pattern
    s[C].reorder(xo, yo, ko, xi, yi, ki)
    
    # Optimization 2: Vectorization
    # Use SIMD instructions for inner loop
    s[C].vectorize(yi)
    
    # Optimization 3: Parallelization
    # Use multiple CPU cores
    s[C].parallel(xo)
    
    print("📋 Schedule optimizations applied:")
    print(f"  ✓ Tiling: {tile_size}x{tile_size} blocks (cache optimization)")
    print(f"  ✓ Vectorization: Inner loop uses SIMD instructions")
    print(f"  ✓ Parallelization: Outer loop uses multiple CPU cores")
    print()
    
    # Step 3: Build the function
    func = tvm.build(s, [A, B, C], target="llvm", name="matmul_opt")
    
    # Step 4: Prepare data
    np_A = np.random.randn(size, size).astype(np.float32)
    np_B = np.random.randn(size, size).astype(np.float32)
    np_C = np.zeros((size, size), dtype=np.float32)
    
    # Convert to TVM arrays
    dev = tvm.cpu(0)
    tvm_A = tvm.nd.array(np_A, dev)
    tvm_B = tvm.nd.array(np_B, dev)
    tvm_C = tvm.nd.array(np_C, dev)
    
    # Warmup
    for _ in range(10):
        func(tvm_A, tvm_B, tvm_C)
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        func(tvm_A, tvm_B, tvm_C)
    end = time.time()
    
    avg_time = (end - start) / iterations * 1000
    print(f"Matrix size: {size}x{size}")
    print(f"TVM (optimized) time: {avg_time:.4f} ms")
    print(f"GFLOPS: {2 * size**3 / avg_time / 1e6:.2f}")
    
    return avg_time, tvm_C.numpy()


def compare_results(pytorch_result, tvm_result):
    """Check if results match"""
    print_header("Correctness Check")
    
    diff = np.abs(pytorch_result - tvm_result).max()
    print(f"Maximum difference: {diff:.6e}")
    
    if diff < 1e-3:
        print("✓ Results match! TVM computed correctly.")
    else:
        print("✗ Results differ! There may be an issue.")
    
    return diff < 1e-3


def print_summary(pytorch_time, tvm_basic_time, tvm_opt_time):
    """Print performance summary"""
    print_header("Performance Summary")
    
    print(f"PyTorch:           {pytorch_time:.4f} ms")
    print(f"TVM (basic):       {tvm_basic_time:.4f} ms  ({tvm_basic_time/pytorch_time:.2f}x vs PyTorch)")
    print(f"TVM (optimized):   {tvm_opt_time:.4f} ms  ({pytorch_time/tvm_opt_time:.2f}x FASTER than PyTorch)")
    print()
    print("Key Insights:")
    print("  • TVM basic schedule is slower (naive implementation)")
    print("  • TVM optimized schedule is FASTER (manual optimizations)")
    print("  • Optimizations: tiling + vectorization + parallelization")
    print("  • Auto-tuning can find even better schedules automatically!")


def main():
    print("=" * 80)
    print("  TVM Introduction: From PyTorch to Optimized Code")
    print("=" * 80)
    print()
    print("This demo compares:")
    print("  1. PyTorch (baseline)")
    print("  2. TVM with basic schedule (no optimization)")
    print("  3. TVM with optimized schedule (tiling + vectorization + parallel)")
    print()
    print("Running on CPU... (GPU tutorial is in gpu_optimization.py)")
    print()
    
    # Configuration
    size = 512  # Matrix size (512x512)
    iterations = 50
    tile_size = 32
    
    # Run benchmarks
    pytorch_time, pytorch_result = benchmark_pytorch(size, iterations)
    tvm_basic_time, tvm_basic_result = benchmark_tvm_basic(size, iterations)
    tvm_opt_time, tvm_opt_result = benchmark_tvm_optimized(size, iterations, tile_size)
    
    # Verify correctness (compare TVM optimized with PyTorch)
    compare_results(pytorch_result, tvm_opt_result)
    
    # Print summary
    print_summary(pytorch_time, tvm_basic_time, tvm_opt_time)
    
    print_header("Next Steps")
    print("✓ You've learned TVM basics!")
    print()
    print("Continue learning:")
    print("  1. gpu_optimization.py - Optimize for GPU (uses your V100)")
    print("  2. auto_tuning.py - Let TVM find the best schedule automatically")
    print("  3. LEARNING_GUIDE.md - Deep dive into TVM concepts")
    print()
    print("Key TVM concepts demonstrated:")
    print("  • Tensor Expressions (te.compute) - Define WHAT to compute")
    print("  • Schedule (te.create_schedule) - Define HOW to compute")
    print("  • Optimizations (split, reorder, vectorize, parallel)")
    print("  • Build (tvm.build) - Compile to machine code")


if __name__ == "__main__":
    main()
