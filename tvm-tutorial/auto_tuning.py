"""
TVM Auto-Tuning Tutorial
=========================

Learn how to use TVM's auto-tuning to find optimal schedules automatically.

This demonstrates:
1. AutoScheduler (Ansor) - Automatic schedule generation
2. MetaSchedule - Latest tuning framework
3. Tuning for your V100 GPU
4. Comparing auto-tuned vs manual schedules

WARNING: Auto-tuning can take 10-60 minutes depending on settings.
         Start with quick_tune=True for a 5-minute demo.
"""

import tvm
from tvm import te, auto_scheduler, relay
import numpy as np
import torch
import time
import os


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def matmul_auto_schedule_demo(M=1024, N=1024, K=1024, quick_tune=True):
    """Demonstrate AutoScheduler for matrix multiplication"""
    print_header("AutoScheduler Demo - Automatic Schedule Search")
    
    # Define computation as a TVM task
    @auto_scheduler.register_workload
    def matmul_auto(m, n, k):
        A = te.placeholder((m, k), name="A")
        B = te.placeholder((k, n), name="B")
        k_axis = te.reduce_axis((0, k), name="k")
        C = te.compute((m, n), lambda i, j: te.sum(A[i, k_axis] * B[k_axis, j], axis=k_axis), name="C")
        return [A, B, C]
    
    # Create target (V100 GPU)
    target = tvm.target.Target("cuda")
    
    # Create the task
    task = tvm.auto_scheduler.SearchTask(
        func=matmul_auto,
        args=(M, N, K),
        target=target
    )
    
    print(f"Task: Matrix multiplication {M}x{K} @ {K}x{N}")
    print(f"Target: {target}")
    print()
    
    # Tuning configuration
    if quick_tune:
        num_trials = 100  # Quick demo - 5-10 minutes
        print("⚡ Quick tune mode: 100 trials (~5-10 minutes)")
    else:
        num_trials = 1000  # Better results - 30-60 minutes
        print("🔥 Full tune mode: 1000 trials (~30-60 minutes)")
    
    print("This will search for the best schedule automatically...")
    print()
    
    # Create log file
    log_file = "matmul_auto_schedule.json"
    
    # Define tuner
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=num_trials,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    
    # Run auto-tuning
    print("Starting auto-tuning... (this may take a while)")
    task.tune(tune_option)
    
    # Load the best schedule
    sch, args = task.apply_best(log_file)
    
    print("\n✓ Auto-tuning complete!")
    print(f"Best schedule saved to: {log_file}")
    print()
    
    # Build the function
    func = tvm.build(sch, args, target)
    
    # Benchmark
    print("Benchmarking auto-tuned kernel...")
    dev = tvm.cuda(0)
    
    np_A = np.random.randn(M, K).astype(np.float32)
    np_B = np.random.randn(K, N).astype(np.float32)
    np_C = np.zeros((M, N), dtype=np.float32)
    
    tvm_A = tvm.nd.array(np_A, dev)
    tvm_B = tvm.nd.array(np_B, dev)
    tvm_C = tvm.nd.array(np_C, dev)
    
    # Warmup
    for _ in range(10):
        func(tvm_A, tvm_B, tvm_C)
    dev.sync()
    
    # Benchmark
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        func(tvm_A, tvm_B, tvm_C)
    dev.sync()
    end = time.time()
    
    avg_time = (end - start) / iterations * 1000
    gflops = (2 * M * N * K) / avg_time / 1e6
    
    print(f"Auto-tuned time: {avg_time:.4f} ms")
    print(f"GFLOPS: {gflops:.2f}")
    
    return avg_time, tvm_C.numpy(), log_file


def compare_with_pytorch(M=1024, N=1024, K=1024):
    """Compare auto-tuned TVM with PyTorch"""
    print_header("PyTorch Baseline")
    
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    
    # Benchmark
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iterations * 1000
    gflops = (2 * M * N * K) / avg_time / 1e6
    
    print(f"PyTorch time: {avg_time:.4f} ms")
    print(f"GFLOPS: {gflops:.2f}")
    
    return avg_time, C.cpu().numpy()


def explain_auto_scheduling():
    """Explain what AutoScheduler does"""
    print_header("How AutoScheduler Works")
    
    print("""
TVM AutoScheduler (Ansor) automatically searches for optimal schedules.

The Search Process:
-------------------
1. **Sketch Generation**: Create high-level schedule templates
   - Example sketches: tiling patterns, memory hierarchies, parallelization

2. **Random Sampling**: Generate candidate schedules from sketches
   - Try different tile sizes (16x16, 32x32, 64x64, etc.)
   - Try different memory scopes (shared, local, global)
   - Try different thread bindings

3. **Measurement**: Run each candidate on your actual V100 GPU
   - Compile the schedule
   - Execute on hardware
   - Measure actual performance

4. **Learning**: Use cost model to predict good schedules
   - Machine learning model predicts performance
   - Guides search to promising regions
   - Avoids testing obviously bad schedules

5. **Evolution**: Iterate and refine
   - Keep best schedules
   - Mutate/combine good schedules
   - Explore new variations

Result:
-------
After 100-1000 trials, AutoScheduler finds schedules that often:
- Match or beat hand-tuned CUDA kernels
- Adapt perfectly to your specific hardware (V100)
- Use optimizations you might not think of manually

Key Optimizations Found:
------------------------
✓ Optimal tile sizes for V100's cache hierarchy
✓ Best shared memory usage patterns
✓ Efficient thread block configurations
✓ Register allocation strategies
✓ Memory access coalescing patterns

This is why auto-tuning takes time but produces excellent results!
""")


def print_final_comparison(pytorch_time, tvm_time, M, N, K):
    """Print final comparison"""
    print_header("Final Results")
    
    total_ops = 2 * M * N * K
    
    print(f"Matrix operation: {M}x{K} @ {K}x{N}")
    print()
    print(f"PyTorch (cuBLAS):  {pytorch_time:.4f} ms  ({total_ops/pytorch_time/1e6:.2f} GFLOPS)")
    print(f"TVM (auto-tuned):  {tvm_time:.4f} ms  ({total_ops/tvm_time/1e6:.2f} GFLOPS)")
    print()
    
    if tvm_time < pytorch_time:
        speedup = pytorch_time / tvm_time
        print(f"🎉 TVM is {speedup:.2f}x FASTER than PyTorch!")
        print()
        print("Impressive! TVM's auto-tuning found a schedule better than cuBLAS.")
    elif tvm_time < pytorch_time * 1.2:
        print(f"⚖️  TVM is competitive with PyTorch (within 20%)")
        print()
        print("Good! For matmul, cuBLAS is highly optimized.")
        print("TVM shines more on custom operations where cuBLAS doesn't apply.")
    else:
        print(f"📊 PyTorch is faster for this operation")
        print()
        print("Note: cuBLAS uses:")
        print("  - Tensor cores (mixed precision)")
        print("  - Years of NVIDIA engineer optimizations")
        print("  - Hardware-specific tuning")
        print()
        print("TVM is more valuable for:")
        print("  - Custom fused operations (where cuBLAS doesn't apply)")
        print("  - Non-NVIDIA hardware (AMD, ARM, custom accelerators)")
        print("  - Operations not in cuBLAS library")
    
    print()
    print("Try these experiments:")
    print("  1. Larger matrices (M=N=K=4096)")
    print("  2. Longer tuning (num_trials=1000)")
    print("  3. Custom fused operations (GELU+matmul)")


def main():
    print("=" * 80)
    print("  TVM Auto-Tuning Tutorial")
    print("=" * 80)
    print()
    print("This tutorial demonstrates automatic schedule optimization with TVM.")
    print()
    print("⚠️  WARNING: Auto-tuning takes time!")
    print("   - Quick mode (100 trials): ~5-10 minutes")
    print("   - Full mode (1000 trials): ~30-60 minutes")
    print()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("✗ CUDA not available. This tutorial requires a GPU.")
        return
    
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    print()
    
    # Configuration
    M, N, K = 1024, 1024, 1024
    quick_tune = True  # Set to False for better results (but longer tuning)
    
    response = input("Start auto-tuning? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Skipping auto-tuning. Set quick_tune=False for better results.")
        explain_auto_scheduling()
        return
    
    # Explain what's happening
    explain_auto_scheduling()
    
    # Run PyTorch baseline
    pytorch_time, pytorch_result = compare_with_pytorch(M, N, K)
    
    # Run auto-tuning
    tvm_time, tvm_result, log_file = matmul_auto_schedule_demo(M, N, K, quick_tune)
    
    # Compare results
    diff = np.abs(pytorch_result - tvm_result).max()
    print_header("Correctness Check")
    print(f"Max difference: {diff:.6e}")
    if diff < 1e-3:
        print("✓ Results match!")
    
    # Final comparison
    print_final_comparison(pytorch_time, tvm_time, M, N, K)
    
    print_header("What's Next?")
    print(f"✓ Auto-tuning log saved to: {log_file}")
    print("✓ You can reuse this log for future runs (no need to re-tune)")
    print()
    print("Advanced topics:")
    print("  - MetaSchedule: Latest tuning framework (TVM 0.10+)")
    print("  - Fused operations: Combine multiple ops for better performance")
    print("  - Tensor cores: Mixed precision for 10x+ speedup")
    print("  - Cross-compilation: Tune once, deploy anywhere")
    print()
    print("See LEARNING_GUIDE.md for deep dive into these topics!")


if __name__ == "__main__":
    main()
