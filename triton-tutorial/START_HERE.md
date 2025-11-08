# Quick Start Guide

## Installation (3 minutes)

```bash
# 1. Install Triton
pip install triton

# 2. Install PyTorch (if you don't have it)
pip install torch torchvision

# 3. Verify installation
python -c "import triton; print(f'Triton {triton.__version__} installed successfully!')"
```

**That's it!** No CUDA toolkit, no cmake, no compilation. Just works.

---

## First Tutorial (10 minutes)

```bash
# Run the softmax fusion tutorial
python simple_fusion.py
```

**Expected output:**
```
🧠 Memory Access Pattern Visualization
================================================================================
... (memory pattern visualization)

🎮 GPU: Tesla V100-SXM2-32GB
💾 Memory: 32.0 GB

================================================================================
Benchmarking Softmax Fusion
Matrix size: 4096×4096
Memory per operation: 67.1 MB
================================================================================

🔥 PyTorch Native Softmax
Time: 0.847ms

⚡ Triton Fused Softmax
Time: 0.281ms

📊 Performance Summary
Speedup: 3.01x 🚀
Memory saved: 33.3%
```

---

## What You'll Learn

### Simple Fusion (`simple_fusion.py`)
- **Problem:** PyTorch's 3 kernel launches for softmax
- **Solution:** Fuse into 1 kernel, keep intermediate values in registers
- **Speedup:** 2-3x
- **Time:** 10-15 minutes to understand

### Layer Normalization (`layer_norm.py`)
- **Problem:** 5 kernel launches for a transformer layer norm
- **Solution:** Two-pass algorithm, everything in registers
- **Speedup:** 1.5-2x
- **Real-world:** Used in EVERY transformer layer
- **Time:** 15-20 minutes

### Flash Attention Lite (`flash_attention_lite.py`)
- **Problem:** Quadratic memory, can't fit long sequences
- **Solution:** Online softmax, block-wise computation
- **Speedup:** 3-4x, plus enables longer sequences
- **Advanced:** Requires understanding of the first two
- **Time:** 30-40 minutes

---

## Learning Path

### Beginner (Today: 1-2 hours)
1. Read `README.md` (this file)
2. Run `python simple_fusion.py`
3. Read the code comments
4. Understand: "Fusion = fewer memory accesses"

### Intermediate (This week: 3-5 hours)
1. Run `python layer_norm.py`
2. Read `LEARNING_GUIDE.md` sections 1-3
3. Modify block sizes and see performance impact
4. Implement a custom element-wise fusion

### Advanced (This month: 10+ hours)
1. Run `python flash_attention_lite.py`
2. Read the Flash Attention paper
3. Read `REAL_WORLD_USES.md` for production patterns
4. Optimize a kernel from your own codebase

---

## Common Issues

### "ModuleNotFoundError: No module named 'triton'"
```bash
pip install triton
```

### "RuntimeError: No CUDA GPUs are available"
You need an NVIDIA GPU. Check:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### "Triton kernel slower than PyTorch!"
- Check block size (try powers of 2: 128, 256, 512, 1024)
- Warmup might not be sufficient (increase warmup iterations)
- Small problem size (Triton shines on larger tensors)

### "Results don't match!"
- Check tolerance (use `torch.allclose(a, b, atol=1e-4)`)
- Floating point precision differs (expected difference ~1e-5)

---

## Next Steps

1. **Run all tutorials** in order
2. **Profile your own models** to find fusion opportunities:
   ```bash
   python -m torch.utils.bottleneck your_script.py
   ```
3. **Read production examples** in `REAL_WORLD_USES.md`
4. **Join the community**:
   - [Triton GitHub Discussions](https://github.com/openai/triton/discussions)
   - [PyTorch Forums](https://discuss.pytorch.org/)

---

## Quick Reference

### Triton Basics
```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate offsets
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute (stays in registers!)
    y = x * 2 + 1
    
    # Store result
    tl.store(y_ptr + offsets, y, mask=mask)

# Launch kernel
grid = (triton.cdiv(n, BLOCK_SIZE),)
my_kernel[grid](x, y, n, BLOCK_SIZE=1024)
```

### Benchmarking Template
```python
import time
import torch

# Warmup
for _ in range(10):
    output = my_function(input)
torch.cuda.synchronize()

# Measure
start = time.perf_counter()
for _ in range(100):
    output = my_function(input)
torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - start) / 100 * 1000
print(f"Time: {elapsed_ms:.3f} ms")
```

---

**Ready?** Run `python simple_fusion.py` and see the magic! 🚀
