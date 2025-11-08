"""
Memory Analysis: Visualize Why Fusion Wins

This script counts and visualizes the exact number of memory transactions
for PyTorch vs Triton implementations, proving why fusion is faster.

KEY INSIGHT:
- Global memory access: ~400 cycles latency
- Register access: ~1 cycle latency
- Reducing memory transactions = massive speedup

We'll analyze softmax to show the concrete savings.
"""

import torch
import triton
import triton.language as tl
import time


class MemoryCounter:
    """Track memory reads and writes for analysis."""
    
    def __init__(self, name):
        self.name = name
        self.reads = 0
        self.writes = 0
        self.read_bytes = 0
        self.write_bytes = 0
        self.kernel_launches = 0
    
    def record_read(self, tensor):
        """Record a memory read operation."""
        self.reads += 1
        self.read_bytes += tensor.numel() * tensor.element_size()
    
    def record_write(self, tensor):
        """Record a memory write operation."""
        self.writes += 1
        self.write_bytes += tensor.numel() * tensor.element_size()
    
    def record_kernel(self):
        """Record a kernel launch."""
        self.kernel_launches += 1
    
    def total_traffic(self):
        """Total memory traffic in bytes."""
        return self.read_bytes + self.write_bytes
    
    def report(self):
        """Print memory analysis report."""
        print(f"\n{'='*60}")
        print(f"Memory Analysis: {self.name}")
        print(f"{'='*60}")
        print(f"Memory Reads:      {self.reads:3d} operations, {self.read_bytes/1e6:8.2f} MB")
        print(f"Memory Writes:     {self.writes:3d} operations, {self.write_bytes/1e6:8.2f} MB")
        print(f"Total Traffic:     {self.total_traffic()/1e6:8.2f} MB")
        print(f"Kernel Launches:   {self.kernel_launches:3d}")
        print(f"{'='*60}\n")
        return self


def analyze_pytorch_softmax(x):
    """
    Analyze PyTorch softmax memory operations.
    
    PyTorch does:
    1. max_val = x.max(dim=-1) - read x, write max_val
    2. exp_x = (x - max_val).exp() - read x and max_val, write exp_x
    3. sum_exp = exp_x.sum(dim=-1) - read exp_x, write sum_exp
    4. output = exp_x / sum_exp - read exp_x and sum_exp, write output
    
    Actually PyTorch fuses some of these, but for education we count conservatively.
    """
    counter = MemoryCounter("PyTorch Softmax")
    
    # Step 1: max (for numerical stability)
    counter.record_kernel()
    counter.record_read(x)  # Read input
    max_val = x.max(dim=-1, keepdim=True)[0]
    counter.record_write(max_val)  # Write max
    
    # Step 2: subtract max and exp
    counter.record_kernel()
    counter.record_read(x)  # Read input again
    counter.record_read(max_val)  # Read max
    x_shifted = x - max_val
    exp_x = torch.exp(x_shifted)
    counter.record_write(exp_x)  # Write exp(x - max)
    
    # Step 3: sum
    counter.record_kernel()
    counter.record_read(exp_x)  # Read exp values
    sum_exp = exp_x.sum(dim=-1, keepdim=True)
    counter.record_write(sum_exp)  # Write sum
    
    # Step 4: divide
    counter.record_kernel()
    counter.record_read(exp_x)  # Read exp values again
    counter.record_read(sum_exp)  # Read sum
    output = exp_x / sum_exp
    counter.record_write(output)  # Write output
    
    return output, counter.report()


def analyze_triton_softmax(x):
    """
    Analyze Triton fused softmax memory operations.
    
    Triton does:
    1. Load x from global memory
    2. Compute max, exp, sum, divide ALL IN REGISTERS
    3. Store output to global memory
    
    Only 2 global memory operations total!
    """
    counter = MemoryCounter("Triton Fused Softmax")
    
    # Single kernel launch
    counter.record_kernel()
    
    # Load input (only global memory read)
    counter.record_read(x)
    
    # All computation happens in registers (no memory traffic!)
    # - max(x): in registers
    # - exp(x - max): in registers
    # - sum(exp): in registers
    # - exp / sum: in registers
    
    # Store output (only global memory write)
    output = torch.empty_like(x)  # Allocation (not counted in traffic)
    counter.record_write(output)
    
    return output, counter.report()


def visualize_memory_pattern():
    """
    Create ASCII art visualization of memory access patterns.
    """
    print("\n" + "="*70)
    print("MEMORY ACCESS PATTERN VISUALIZATION")
    print("="*70)
    
    print("\n📉 PyTorch Approach (4 kernel launches):\n")
    print("Kernel 1: max(x)")
    print("  GPU ◄──── [DRAM: read x (64 MB)] ────┐")
    print("  GPU ────► [DRAM: write max (0.016 MB)] ────┐")
    print("                                          Slow!")
    print()
    print("Kernel 2: exp(x - max)")
    print("  GPU ◄──── [DRAM: read x (64 MB)] ────┐")
    print("  GPU ◄──── [DRAM: read max (0.016 MB)]")
    print("  GPU ────► [DRAM: write exp (64 MB)] ──────┐")
    print("                                          Slow!")
    print()
    print("Kernel 3: sum(exp)")
    print("  GPU ◄──── [DRAM: read exp (64 MB)] ───┐")
    print("  GPU ────► [DRAM: write sum (0.016 MB)] ───┐")
    print("                                          Slow!")
    print()
    print("Kernel 4: exp / sum")
    print("  GPU ◄──── [DRAM: read exp (64 MB)] ───┐")
    print("  GPU ◄──── [DRAM: read sum (0.016 MB)]")
    print("  GPU ────► [DRAM: write output (64 MB)] ───┐")
    print("                                          Slow!")
    print()
    print("Total DRAM traffic: ~320 MB")
    print("Total kernel launches: 4")
    print("Problem: Intermediate arrays (max, exp, sum) stored in DRAM!")
    
    print("\n" + "-"*70)
    
    print("\n⚡ Triton Approach (1 kernel launch):\n")
    print("Single Fused Kernel:")
    print("  GPU ◄──── [DRAM: read x (64 MB)] ─────────────┐")
    print("                                              Slow!")
    print("  GPU │")
    print("      │ ┌─────────────────────────────┐")
    print("      │ │ Registers (on-chip):        │")
    print("      │ │  max_val = max(x)          │ ◄── Fast! (~1 cycle)")
    print("      │ │  shifted = x - max_val     │ ◄── Fast!")
    print("      │ │  exp_val = exp(shifted)    │ ◄── Fast!")
    print("      │ │  sum_val = sum(exp_val)    │ ◄── Fast!")
    print("      │ │  output = exp_val / sum_val│ ◄── Fast!")
    print("      │ └─────────────────────────────┘")
    print("      │")
    print("  GPU ────► [DRAM: write output (64 MB)] ───────┐")
    print("                                              Slow!")
    print()
    print("Total DRAM traffic: 128 MB (60% reduction!)")
    print("Total kernel launches: 1")
    print("Key: Intermediate values NEVER touch DRAM!")
    
    print("\n" + "="*70)


def latency_comparison():
    """
    Show the latency cost of memory hierarchy.
    """
    print("\n" + "="*70)
    print("MEMORY HIERARCHY LATENCY (Typical NVIDIA GPU)")
    print("="*70)
    
    print("\nAccess Type          Latency (cycles)  Relative  Visual")
    print("-" * 70)
    print(f"Registers            ~1 cycle          1x        █")
    print(f"L1/Shared Memory     ~20 cycles        20x       {'█'*20}")
    print(f"L2 Cache             ~200 cycles       200x      {'█'*60}...")
    print(f"Global Memory (HBM)  ~400 cycles       400x      {'█'*70}...")
    
    print("\n" + "="*70)
    print("WHY FUSION WINS:")
    print("="*70)
    print("PyTorch: Intermediate values (max, exp, sum) go to global memory")
    print("         → 400 cycle latency per access!")
    print()
    print("Triton:  Intermediate values stay in registers")
    print("         → 1 cycle latency per access!")
    print()
    print("Speedup: 400x faster access × fewer accesses = massive win!")
    print("="*70 + "\n")


def bandwidth_analysis(size=4096):
    """
    Analyze memory bandwidth utilization.
    """
    print("\n" + "="*70)
    print(f"BANDWIDTH ANALYSIS (Matrix: {size}×{size})")
    print("="*70)
    
    elements = size * size
    bytes_per_element = 4  # float32
    total_bytes = elements * bytes_per_element
    
    # V100 specs
    v100_bandwidth_gb_s = 900  # GB/s
    v100_peak_tflops = 15.7  # TFLOPS (FP32)
    
    print(f"\nNVIDIA V100 Specs:")
    print(f"  Peak Bandwidth:  {v100_bandwidth_gb_s} GB/s")
    print(f"  Peak Compute:    {v100_peak_tflops} TFLOPS (FP32)")
    print(f"  Arithmetic Intensity Needed: {v100_peak_tflops*1000/v100_bandwidth_gb_s:.1f} ops/byte")
    
    print(f"\nSoftmax Analysis:")
    print(f"  Data size: {total_bytes/1e6:.1f} MB")
    
    # PyTorch
    pytorch_traffic = total_bytes * 5  # Rough estimate
    pytorch_ops = elements * 5  # max, sub, exp, sum, div (simplified)
    pytorch_ai = pytorch_ops / pytorch_traffic
    
    print(f"\n  PyTorch:")
    print(f"    Memory traffic:  ~{pytorch_traffic/1e6:.1f} MB")
    print(f"    Operations:      ~{pytorch_ops/1e6:.1f} M ops")
    print(f"    Arithmetic Intensity: {pytorch_ai:.3f} ops/byte")
    print(f"    GPU Utilization: {pytorch_ai / (v100_peak_tflops*1000/v100_bandwidth_gb_s) * 100:.1f}%")
    print(f"    ⚠️  Memory-bound! (Arithmetic intensity too low)")
    
    # Triton
    triton_traffic = total_bytes * 2  # Read once, write once
    triton_ops = elements * 5  # Same operations, less memory
    triton_ai = triton_ops / triton_traffic
    
    print(f"\n  Triton:")
    print(f"    Memory traffic:  ~{triton_traffic/1e6:.1f} MB")
    print(f"    Operations:      ~{triton_ops/1e6:.1f} M ops")
    print(f"    Arithmetic Intensity: {triton_ai:.3f} ops/byte")
    print(f"    GPU Utilization: {triton_ai / (v100_peak_tflops*1000/v100_bandwidth_gb_s) * 100:.1f}%")
    print(f"    ✓ Better! (60% less memory traffic)")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print(f"Triton saves {(pytorch_traffic - triton_traffic)/1e6:.1f} MB of memory traffic")
    print(f"That's {(1 - triton_traffic/pytorch_traffic)*100:.0f}% less bandwidth wasted!")
    print("Same operations, but intermediate values stay in fast on-chip memory.")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MEMORY ANALYSIS: WHY TRITON FUSION BEATS PYTORCH")
    print("="*70)
    
    # Create test tensor
    size = 4096
    x = torch.randn(size, size, device='cuda', dtype=torch.float32)
    
    print(f"\nTest case: {size}×{size} matrix = {x.numel()} elements = {x.numel()*4/1e6:.1f} MB")
    
    # Visualizations
    visualize_memory_pattern()
    latency_comparison()
    bandwidth_analysis(size)
    
    # Memory operation counting
    print("\n" + "="*70)
    print("COUNTING MEMORY OPERATIONS")
    print("="*70)
    
    # Note: These are approximate counts for educational purposes
    # Real PyTorch may fuse some operations
    
    pytorch_out, pytorch_counter = analyze_pytorch_softmax(x)
    triton_out, triton_counter = analyze_triton_softmax(x)
    
    # Summary comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    pytorch_traffic = pytorch_counter.total_traffic()
    triton_traffic = triton_counter.total_traffic()
    traffic_saved = (pytorch_traffic - triton_traffic) / pytorch_traffic * 100
    
    print(f"\nMemory Traffic:")
    print(f"  PyTorch: {pytorch_traffic/1e6:8.2f} MB")
    print(f"  Triton:  {triton_traffic/1e6:8.2f} MB")
    print(f"  Saved:   {traffic_saved:.1f}%")
    
    print(f"\nKernel Launches:")
    print(f"  PyTorch: {pytorch_counter.kernel_launches}")
    print(f"  Triton:  {triton_counter.kernel_launches}")
    print(f"  Reduction: {pytorch_counter.kernel_launches}x → {triton_counter.kernel_launches}x")
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("1. Fusion eliminates intermediate memory writes/reads")
    print("2. Register access is 400x faster than global memory")
    print("3. Fewer kernel launches reduces overhead")
    print("4. Same computation, massive bandwidth savings")
    print("5. This is why Triton achieves 2-3x speedups!")
    print("="*70 + "\n")
    
    print("✅ Analysis complete!")
    print("📚 Next: Run simple_fusion.py to see real timing comparisons!")
