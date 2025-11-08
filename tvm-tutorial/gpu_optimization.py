"""
TVM GPU Optimization Tutorial
==============================

Learn how to optimize GPU kernels with TVM on your Tesla V100.

This demonstrates:
1. Basic GPU schedule
2. Optimized GPU schedule (thread binding, shared memory)
3. Comparison with PyTorch CUDA
4. Understanding TVM's GPU optimizations

Target: NVIDIA Tesla V100-SXM2-32GB (Compute Capability 7.0)
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


def check_gpu():
    """Check if CUDA GPU is available"""
    print_header("GPU Check")
    
    # Check PyTorch CUDA
    if not torch.cuda.is_available():
        print("✗ PyTorch CUDA not available!")
        return False
    
    print(f"✓ PyTorch CUDA available")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    
    # Check TVM CUDA
    try:
        dev = tvm.cuda(0)
        print(f"✓ TVM CUDA available")
        return True
    except:
        print("✗ TVM CUDA not available!")
        return False


def benchmark_pytorch_gpu(M=4096, N=4096, K=4096, iterations=100):
    """Benchmark PyTorch GPU matrix multiplication"""
    print_header(f"PyTorch CUDA Baseline - {M}x{K} @ {K}x{N}")
    
    # Create random matrices on GPU
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iterations * 1000  # Convert to ms
    gflops = (2 * M * N * K) / avg_time / 1e6
    
    print(f"Matrix sizes: {M}x{K} @ {K}x{N} = {M}x{N}")
    print(f"PyTorch time: {avg_time:.4f} ms")
    print(f"GFLOPS: {gflops:.2f}")
    print(f"Memory accessed: {(M*K + K*N + M*N) * 4 / 1e9:.3f} GB")
    
    return avg_time, C.cpu().numpy()


def benchmark_tvm_gpu_basic(M=4096, N=4096, K=4096, iterations=100):
    """Benchmark TVM GPU with basic schedule"""
    print_header(f"TVM CUDA - Basic Schedule - {M}x{K} @ {K}x{N}")
    
    # Define computation
    m, n, k = te.var("m"), te.var("n"), te.var("k")
    A = te.placeholder((m, k), name="A", dtype="float32")
    B = te.placeholder((k, n), name="B", dtype="float32")
    k_axis = te.reduce_axis((0, k), name="k")
    C = te.compute((m, n), lambda i, j: te.sum(A[i, k_axis] * B[k_axis, j], axis=k_axis), name="C")
    
    # Create basic GPU schedule
    s = te.create_schedule(C.op)
    
    # Just bind to GPU threads - no optimization
    x, y = C.op.axis
    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis("threadIdx.x")
    
    xo, xi = s[C].split(x, factor=16)
    s[C].bind(xo, block_x)
    s[C].bind(xi, thread_x)
    
    # Build for CUDA
    func = tvm.build(s, [A, B, C], target="cuda", name="matmul_gpu_basic")
    
    # Prepare data
    np_A = np.random.randn(M, K).astype(np.float32)
    np_B = np.random.randn(K, N).astype(np.float32)
    np_C = np.zeros((M, N), dtype=np.float32)
    
    # Convert to TVM arrays on GPU
    dev = tvm.cuda(0)
    tvm_A = tvm.nd.array(np_A, dev)
    tvm_B = tvm.nd.array(np_B, dev)
    tvm_C = tvm.nd.array(np_C, dev)
    
    # Warmup
    for _ in range(10):
        func(tvm_A, tvm_B, tvm_C)
    dev.sync()
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        func(tvm_A, tvm_B, tvm_C)
    dev.sync()
    end = time.time()
    
    avg_time = (end - start) / iterations * 1000
    gflops = (2 * M * N * K) / avg_time / 1e6
    
    print(f"TVM (basic) time: {avg_time:.4f} ms")
    print(f"GFLOPS: {gflops:.2f}")
    print("Note: Basic schedule is naive - expect poor performance")
    
    return avg_time, tvm_C.numpy()


def benchmark_tvm_gpu_optimized(M=4096, N=4096, K=4096, iterations=100):
    """Benchmark TVM GPU with optimized schedule"""
    print_header(f"TVM CUDA - Optimized Schedule - {M}x{K} @ {K}x{N}")
    
    # Define computation
    m, n, k = te.var("m"), te.var("n"), te.var("k")
    A = te.placeholder((m, k), name="A", dtype="float32")
    B = te.placeholder((k, n), name="B", dtype="float32")
    k_axis = te.reduce_axis((0, k), name="k")
    C = te.compute((m, n), lambda i, j: te.sum(A[i, k_axis] * B[k_axis, j], axis=k_axis), name="C")
    
    # Create OPTIMIZED GPU schedule
    s = te.create_schedule(C.op)
    
    # Configuration for V100
    tile_x, tile_y = 64, 64
    tile_k = 8
    
    # Cache for shared memory
    AA = s.cache_read(A, "shared", [C])
    BB = s.cache_read(B, "shared", [C])
    CC = s.cache_write(C, "local")
    
    # Get axes
    x, y = C.op.axis
    k_outer, k_inner = s[C].split(C.op.reduce_axis[0], factor=tile_k)
    
    # Tile the output
    xo, xi = s[C].split(x, factor=tile_x)
    yo, yi = s[C].split(y, factor=tile_y)
    
    # Reorder for better memory access
    s[C].reorder(xo, yo, k_outer, k_inner, xi, yi)
    
    # Bind to GPU threads
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    
    s[C].bind(xo, block_x)
    s[C].bind(yo, block_y)
    
    # Further split for thread binding
    xii, xiii = s[C].split(xi, factor=8)
    yii, yiii = s[C].split(yi, factor=8)
    s[C].bind(xii, thread_x)
    s[C].bind(yii, thread_y)
    
    # Compute cache writes
    s[CC].compute_at(s[C], yii)
    
    # Cooperative fetching for shared memory
    s[AA].compute_at(s[C], k_outer)
    s[BB].compute_at(s[C], k_outer)
    
    # Thread binding for shared memory loads
    x_aa, y_aa = s[AA].op.axis
    xo_aa, xi_aa = s[AA].split(x_aa, factor=8)
    yo_aa, yi_aa = s[AA].split(y_aa, factor=8)
    s[AA].bind(xo_aa, thread_x)
    s[AA].bind(yo_aa, thread_y)
    
    x_bb, y_bb = s[BB].op.axis
    xo_bb, xi_bb = s[BB].split(x_bb, factor=8)
    yo_bb, yi_bb = s[BB].split(y_bb, factor=8)
    s[BB].bind(xo_bb, thread_x)
    s[BB].bind(yo_bb, thread_y)
    
    print("📋 GPU optimizations applied:")
    print(f"  ✓ Tiling: {tile_x}x{tile_y} output tiles")
    print(f"  ✓ Shared memory: Cached A and B in shared memory")
    print(f"  ✓ Local memory: Cached C writes in registers")
    print(f"  ✓ Thread binding: 8x8 thread blocks")
    print(f"  ✓ Cooperative fetching: Threads load shared memory together")
    print(f"  ✓ K-dimension tiling: {tile_k} for reduction")
    print()
    
    # Build for CUDA
    func = tvm.build(s, [A, B, C], target="cuda", name="matmul_gpu_opt")
    
    # Prepare data
    np_A = np.random.randn(M, K).astype(np.float32)
    np_B = np.random.randn(K, N).astype(np.float32)
    np_C = np.zeros((M, N), dtype=np.float32)
    
    # Convert to TVM arrays on GPU
    dev = tvm.cuda(0)
    tvm_A = tvm.nd.array(np_A, dev)
    tvm_B = tvm.nd.array(np_B, dev)
    tvm_C = tvm.nd.array(np_C, dev)
    
    # Warmup
    for _ in range(10):
        func(tvm_A, tvm_B, tvm_C)
    dev.sync()
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        func(tvm_A, tvm_B, tvm_C)
    dev.sync()
    end = time.time()
    
    avg_time = (end - start) / iterations * 1000
    gflops = (2 * M * N * K) / avg_time / 1e6
    
    print(f"TVM (optimized) time: {avg_time:.4f} ms")
    print(f"GFLOPS: {gflops:.2f}")
    
    return avg_time, tvm_C.numpy()


def compare_results(pytorch_result, tvm_result, name="TVM"):
    """Check if results match"""
    print_header(f"Correctness Check - {name}")
    
    diff = np.abs(pytorch_result - tvm_result).max()
    relative_diff = diff / np.abs(pytorch_result).max()
    
    print(f"Maximum absolute difference: {diff:.6e}")
    print(f"Maximum relative difference: {relative_diff:.6e}")
    
    if relative_diff < 1e-3:
        print(f"✓ Results match! {name} computed correctly.")
        return True
    else:
        print(f"✗ Results differ! There may be an issue with {name}.")
        return False


def print_summary(pytorch_time, tvm_basic_time, tvm_opt_time, M, N, K):
    """Print performance summary"""
    print_header("Performance Summary")
    
    total_ops = 2 * M * N * K
    
    print(f"Matrix operation: {M}x{K} @ {K}x{N} = {M}x{N}")
    print(f"Total FLOPs: {total_ops / 1e9:.2f} GFLOPs")
    print()
    print(f"PyTorch CUDA:      {pytorch_time:.4f} ms  ({total_ops/pytorch_time/1e6:.2f} GFLOPS)")
    print(f"TVM (basic):       {tvm_basic_time:.4f} ms  ({total_ops/tvm_basic_time/1e6:.2f} GFLOPS) - {pytorch_time/tvm_basic_time:.2f}x vs PyTorch")
    print(f"TVM (optimized):   {tvm_opt_time:.4f} ms  ({total_ops/tvm_opt_time/1e6:.2f} GFLOPS) - {pytorch_time/tvm_opt_time:.2f}x vs PyTorch")
    print()
    
    if tvm_opt_time < pytorch_time:
        print(f"🎉 TVM optimized is {pytorch_time/tvm_opt_time:.2f}x FASTER than PyTorch!")
    else:
        print(f"Note: PyTorch uses highly optimized cuBLAS library")
        print(f"      For matmul, cuBLAS is hard to beat without tensor cores")
        print(f"      TVM shines more on custom/fused operations")
    
    print()
    print("Key Insights:")
    print("  • PyTorch uses cuBLAS (NVIDIA's optimized BLAS library)")
    print("  • TVM basic schedule is naive (poor memory access patterns)")
    print("  • TVM optimized schedule uses:")
    print("    - Shared memory for data reuse")
    print("    - Register blocking for accumulation")
    print("    - Thread cooperation for memory loads")
    print("  • TVM can match/beat cuBLAS with tensor cores (see auto_tuning.py)")


def main():
    print("=" * 80)
    print("  TVM GPU Optimization Tutorial - Tesla V100")
    print("=" * 80)
    print()
    print("This demo shows how TVM optimizes GPU kernels.")
    print("We'll compare:")
    print("  1. PyTorch CUDA (cuBLAS baseline)")
    print("  2. TVM with basic GPU schedule")
    print("  3. TVM with optimized GPU schedule")
    print()
    
    # Check GPU availability
    if not check_gpu():
        print("GPU not available. Please run simple_intro.py for CPU tutorial.")
        return
    
    # Configuration - smaller sizes for faster testing
    # For real benchmarks, try M=N=K=8192
    M, N, K = 2048, 2048, 2048
    iterations = 50
    
    print(f"\nRunning benchmarks with matrix size: {M}x{K} @ {K}x{N}")
    print(f"(Use M=N=K=8192 for more realistic comparison, but it takes longer)")
    print()
    
    # Run benchmarks
    pytorch_time, pytorch_result = benchmark_pytorch_gpu(M, N, K, iterations)
    tvm_basic_time, tvm_basic_result = benchmark_tvm_gpu_basic(M, N, K, iterations)
    tvm_opt_time, tvm_opt_result = benchmark_tvm_gpu_optimized(M, N, K, iterations)
    
    # Verify correctness
    compare_results(pytorch_result, tvm_opt_result, "TVM Optimized")
    
    # Print summary
    print_summary(pytorch_time, tvm_basic_time, tvm_opt_time, M, N, K)
    
    print_header("Next Steps")
    print("✓ You've learned GPU optimization with TVM!")
    print()
    print("Continue learning:")
    print("  1. auto_tuning.py - Let TVM auto-tune for peak performance")
    print("  2. Try larger matrices (M=N=K=8192) for realistic comparison")
    print("  3. LEARNING_GUIDE.md - Deep dive into TVM GPU concepts")
    print()
    print("Want even better performance?")
    print("  • Auto-tuning can find better schedules than manual optimization")
    print("  • Tensor Cores (on V100) can achieve 100+ TFLOPS for mixed precision")
    print("  • See auto_tuning.py for automatic schedule search")


if __name__ == "__main__":
    main()
