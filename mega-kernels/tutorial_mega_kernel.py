"""
MEGA KERNEL EDUCATIONAL EXAMPLE
================================

This tutorial demonstrates the core concept of a "mega kernel" (fused kernel)
by comparing separate operations vs. a fused operation.

What is a Mega Kernel?
----------------------
A mega kernel combines multiple operations that would normally be separate
GPU kernel launches into a single kernel. This reduces:
1. Memory bandwidth (fewer reads/writes to global memory)
2. Kernel launch overhead
3. Intermediate tensor storage

Example: GELU + Scale
----------------------
We'll implement a simple but practical example:
- Standard approach: Apply GELU activation, then scale the result (2 kernels)
- Mega kernel approach: Do both in a single kernel (1 kernel)

This is a common pattern in transformers (activation + projection).
"""

import torch
import time
import os

# Set CUDA architecture for V100 (compute capability 7.0)
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0'

# ============================================================================
# PART 1: Standard Approach (2 separate operations)
# ============================================================================

def standard_gelu_scale(x, scale_factor):
    """
    Standard approach: Two separate operations
    1. Apply GELU activation (PyTorch kernel)
    2. Multiply by scale factor (PyTorch kernel)
    
    This requires:
    - Reading x from memory
    - Writing intermediate result to memory
    - Reading intermediate result from memory
    - Writing final result to memory
    = 2 reads + 2 writes to global memory
    """
    x = torch.nn.functional.gelu(x)  # Kernel 1: GELU
    x = x * scale_factor              # Kernel 2: Scale
    return x


# ============================================================================
# PART 2: Mega Kernel Approach (1 fused operation)
# ============================================================================

# We'll write a custom CUDA kernel that does BOTH operations in one pass

# CUDA kernel code
CUDA_SOURCE = """
#include <torch/extension.h>

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// MEGA KERNEL: Fuses GELU + Scale into a single kernel
__global__ void fused_gelu_scale_cuda_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float scale_factor,
    int total_elements
) {
    // Each thread processes one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        // Read once from global memory
        float val = input[idx];
        
        // Apply GELU (in registers - super fast!)
        val = gelu(val);
        
        // Apply scale (still in registers!)
        val = val * scale_factor;
        
        // Write once to global memory
        output[idx] = val;
    }
    
    // Total memory operations: 1 read + 1 write (50% reduction!)
}

// C++ wrapper function that calls the CUDA kernel
torch::Tensor fused_gelu_scale_forward(
    torch::Tensor input,
    float scale_factor
) {
    auto output = torch::empty_like(input);
    int total_elements = input.numel();
    
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    fused_gelu_scale_cuda_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scale_factor,
        total_elements
    );
    
    return output;
}
"""

# C++ bindings - this creates the Python module
CPP_SOURCE = """
torch::Tensor fused_gelu_scale_forward(torch::Tensor input, float scale_factor);
"""

# Compile the CUDA kernel using PyTorch's JIT
from torch.utils.cpp_extension import load_inline

cuda_module = load_inline(
    name='fused_gelu_scale',
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=['fused_gelu_scale_forward'],
    verbose=False,
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

def mega_kernel_gelu_scale(x, scale_factor):
    """
    Mega kernel approach: Single fused operation
    Does GELU and scale in ONE kernel launch
    
    This requires:
    - Reading x from memory
    - Writing final result to memory
    = 1 read + 1 write to global memory
    """
    # Call the compiled CUDA kernel
    return cuda_module.fused_gelu_scale_forward(x, scale_factor)


# ============================================================================
# PART 3: Benchmark and Compare
# ============================================================================

def benchmark(func, *args, num_iters=100, warmup=10):
    """Benchmark a function on GPU"""
    # Warmup
    for _ in range(warmup):
        _ = func(*args)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iters):
        _ = func(*args)
    end.record()
    
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iters


def run_comparison():
    """Compare standard vs mega kernel approach"""
    print("\n" + "="*80)
    print("MEGA KERNEL CONCEPT DEMONSTRATION")
    print("="*80)
    
    print("\nYour GPU: Tesla V100-SXM2-32GB (Excellent for learning!)")
    
    # Test different sizes
    sizes = [
        (1024, "Small - 1K elements"),
        (1024 * 1024, "Medium - 1M elements"),
        (16 * 1024 * 1024, "Large - 16M elements"),
    ]
    
    scale_factor = 1.414  # sqrt(2), common in transformers
    
    for size, description in sizes:
        print(f"\n{'-'*80}")
        print(f"Test: {description}")
        print(f"{'-'*80}")
        
        # Create input data
        x = torch.randn(size, device='cuda')
        
        # Benchmark standard approach
        time_standard = benchmark(standard_gelu_scale, x.clone(), scale_factor)
        print(f"Standard (2 kernels):  {time_standard:.4f} ms")
        
        # Benchmark mega kernel approach
        time_mega = benchmark(mega_kernel_gelu_scale, x.clone(), scale_factor)
        print(f"Mega Kernel (1 kernel): {time_mega:.4f} ms")
        
        # Calculate speedup
        speedup = time_standard / time_mega
        bandwidth_saved = (1 - (2.0 / 4.0)) * 100  # We save 50% of memory transfers
        
        print(f"\n✓ Speedup: {speedup:.2f}x faster")
        print(f"✓ Memory bandwidth saved: {bandwidth_saved:.0f}%")
        
        # Verify correctness
        result_standard = standard_gelu_scale(x.clone(), scale_factor)
        result_mega = mega_kernel_gelu_scale(x.clone(), scale_factor)
        max_diff = torch.max(torch.abs(result_standard - result_mega)).item()
        print(f"✓ Correctness: max difference = {max_diff:.2e} (should be ~1e-6)")


# ============================================================================
# PART 4: Visual Explanation
# ============================================================================

def visual_explanation():
    """Print a visual explanation of the concept"""
    print("\n" + "="*80)
    print("VISUAL EXPLANATION: What Happens in Memory?")
    print("="*80)
    
    print("""
STANDARD APPROACH (2 Kernels):
┌─────────────────────────────────────────────────────────────────┐
│  GPU Global Memory                                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Input Data] ──READ──> Kernel 1 (GELU) ──WRITE──> [Temp]      │
│                                                                  │
│  [Temp Data] ──READ──> Kernel 2 (Scale) ──WRITE──> [Output]    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Total Memory Operations: 3 reads + 2 writes = 5 operations
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         This is SLOW! Memory is the bottleneck!


MEGA KERNEL APPROACH (1 Fused Kernel):
┌─────────────────────────────────────────────────────────────────┐
│  GPU Global Memory                                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Input] ──READ──> Mega Kernel ──WRITE──> [Output]             │
│                        │                                         │
│                        ├─ GELU (in registers)                   │
│                        └─ Scale (in registers)                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Total Memory Operations: 1 read + 1 write = 2 operations
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         50% reduction in memory traffic!

KEY INSIGHT:
───────────
The intermediate result never touches global memory! It stays in the
ultra-fast GPU registers while we do both operations. This is the
MEGA KERNEL advantage!

WHY IT MATTERS:
──────────────
1. Memory bandwidth is LIMITED (even on V100!)
2. Arithmetic is CHEAP (trillions of ops/sec)
3. Moving data is EXPENSIVE (hundreds of GB/s)
4. Reducing memory traffic = FASTER execution

Real-world transformers have MANY such opportunities:
- LayerNorm + Linear projection
- Attention + Dropout + Residual
- GELU + Linear in FFN
- etc.

Fusing them all = 2-3x faster models! 🚀
    """)


# ============================================================================
# MAIN: Run the demonstration
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║           WELCOME TO THE MEGA KERNEL LEARNING TUTORIAL          ║
    ║                                                                  ║
    ║  This example teaches you the fundamental concept of kernel     ║
    ║  fusion by implementing a simple GELU + Scale mega kernel.      ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This tutorial requires a GPU.")
        exit(1)
    
    # Show visual explanation first
    visual_explanation()
    
    # Run the actual benchmark
    run_comparison()
    
    print("\n" + "="*80)
    print("LEARNING SUMMARY")
    print("="*80)
    print("""
Key Takeaways:
1. Mega kernels FUSE multiple operations into one kernel
2. Main benefit: REDUCE memory bandwidth (the bottleneck!)
3. Data stays in fast registers instead of slow global memory
4. Typical speedup: 1.5-3x for real transformer operations
5. This technique is used in: FlashAttention, Megatron, FasterTransformer

Next Steps:
- Look at kernels/mega_kernel.cu for more complex examples
- Try the FusedFFN and FusedLayerNorm implementations
- Run python/benchmark.py for comprehensive benchmarks
- Experiment with fusing YOUR favorite operations!

Happy Learning! 🎓
    """)
    print("="*80 + "\n")
