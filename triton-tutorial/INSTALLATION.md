# Installation Guide

## 🚀 Quick Install (Recommended)

```bash
# Install Triton (seriously, that's it!)
pip install triton

# Optional: Install PyTorch for comparisons
pip install torch torchvision

# Verify installation
python -c "import triton; print(f'Triton {triton.__version__} installed!')"
```

**Total time: 2-3 minutes**

---

## ✅ Requirements

### Hardware
- **NVIDIA GPU** with compute capability 7.0 or higher:
  - ✅ V100, T4, A100, H100
  - ✅ RTX 2000 series (2060, 2070, 2080, 2080 Ti)
  - ✅ RTX 3000 series (3060, 3070, 3080, 3090)
  - ✅ RTX 4000 series (4060, 4070, 4080, 4090)
  - ❌ GTX 1000 series (too old, compute capability 6.x)

**Check your GPU:**
```bash
nvidia-smi
```

**Check compute capability:**
```python
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    if props.major >= 7:
        print("✅ Compatible with Triton!")
    else:
        print("❌ Too old for Triton (need compute 7.0+)")
else:
    print("❌ No CUDA GPU found")
```

### Software
- **Python 3.8+** (3.9, 3.10, 3.11, 3.12 supported)
- **CUDA drivers** (usually pre-installed with PyTorch)
- **No CUDA toolkit needed!** (Triton bundles everything)

---

## 📦 Installation Methods

### Method 1: pip (Recommended)

```bash
# Install latest stable version
pip install triton

# Or specify version
pip install triton==2.1.0
```

### Method 2: With PyTorch

If you don't have PyTorch yet:

```bash
# CUDA 11.8
pip install torch torchvision triton

# CUDA 12.1
pip install torch torchvision triton --index-url https://download.pytorch.org/whl/cu121
```

### Method 3: Conda

```bash
# Create environment
conda create -n triton-tutorial python=3.10
conda activate triton-tutorial

# Install packages
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install triton
```

---

## 🧪 Verify Installation

### Test 1: Import Test
```bash
python -c "import triton; print('Triton:', triton.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

**Expected output:**
```
Triton: 2.1.0
PyTorch: 2.1.0+cu121
CUDA: True
```

### Test 2: Simple Kernel
```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

# Test
x = torch.randn(1024, device='cuda')
y = torch.randn(1024, device='cuda')
output = torch.empty_like(x)

grid = lambda meta: (triton.cdiv(1024, meta['BLOCK_SIZE']),)
add_kernel[grid](x, y, output, 1024, BLOCK_SIZE=256)

# Verify
expected = x + y
assert torch.allclose(output, expected), "❌ Test failed!"
print("✅ Triton kernel works!")
```

### Test 3: Run Tutorial
```bash
cd triton-tutorial
python simple_fusion.py
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'triton'"

**Solution:**
```bash
pip install triton
```

### Issue: "RuntimeError: No CUDA GPUs are available"

**Possible causes:**
1. No NVIDIA GPU in system
2. CUDA drivers not installed
3. Wrong PyTorch version (CPU-only)

**Check:**
```bash
# Check GPU visibility
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Issue: "Triton kernel is slower than PyTorch!"

**Possible causes:**
1. Problem size too small (overhead dominates)
2. Block size not optimal
3. Not enough warmup iterations

**Solutions:**
```python
# Increase problem size
x = torch.randn(8192, 8192, device='cuda')  # Larger tensor

# Try different block sizes
for BLOCK_SIZE in [128, 256, 512, 1024, 2048]:
    # Benchmark with this BLOCK_SIZE
    ...

# More warmup
for _ in range(20):  # More warmup iterations
    result = kernel(input)
torch.cuda.synchronize()
```

### Issue: "CUDA out of memory"

**Solutions:**
```python
# Reduce problem size
x = torch.randn(2048, 2048, device='cuda')  # Smaller

# Or free memory
torch.cuda.empty_cache()

# Check memory usage
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Issue: "Numerical differences between PyTorch and Triton"

**Expected behavior:**
- Floating point arithmetic is not perfectly deterministic
- Small differences (< 1e-5) are normal
- Use tolerances in comparisons:

```python
# Good
assert torch.allclose(triton_out, pytorch_out, atol=1e-4, rtol=1e-5)

# Bad (too strict)
assert torch.equal(triton_out, pytorch_out)
```

### Issue: Installation hangs on "Building wheels"

**Solution:**
```bash
# Use pre-built wheels
pip install --prefer-binary triton

# Or specify exact version
pip install triton==2.1.0
```

---

## 🎯 Platform-Specific Notes

### Windows
- ✅ Supported (with WSL2 or native CUDA)
- Install Visual Studio Build Tools if building from source
- Recommend WSL2 for best experience

### Linux
- ✅ Fully supported (best platform)
- Most tested configuration
- No special requirements

### macOS
- ❌ Not supported (no NVIDIA GPU support on modern Macs)
- For M1/M2 Macs, use PyTorch MPS backend instead

---

## 📊 Version Compatibility

| Triton | PyTorch | CUDA | Python | Status |
|--------|---------|------|--------|--------|
| 2.1.0  | 2.0+    | 11.8, 12.1 | 3.8-3.12 | ✅ Recommended |
| 2.0.0  | 1.13+   | 11.7, 11.8 | 3.8-3.11 | ✅ Stable |
| 1.1.1  | 1.12+   | 11.3-11.7  | 3.7-3.10 | ⚠️ Old |

**Recommendation:** Use Triton 2.1.0 + PyTorch 2.0+ for best performance.

---

## 🔧 Advanced Setup

### Development Install (for Triton contributors)

```bash
git clone https://github.com/openai/triton.git
cd triton/python
pip install -e .
```

### Docker Setup

```dockerfile
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install torch torchvision triton

WORKDIR /workspace
COPY triton-tutorial/ /workspace/

CMD ["python3", "simple_fusion.py"]
```

Build and run:
```bash
docker build -t triton-tutorial .
docker run --gpus all triton-tutorial
```

---

## ✅ Post-Installation Checklist

- [ ] `import triton` works
- [ ] `import torch` works
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] GPU has compute capability >= 7.0
- [ ] `python simple_fusion.py` runs successfully
- [ ] Speedup > 1.0x observed

If all checks pass: **You're ready to learn Triton! 🚀**

---

## 🆘 Still Having Issues?

1. **Check GitHub Issues:** [github.com/openai/triton/issues](https://github.com/openai/triton/issues)
2. **Ask in Discussions:** [github.com/openai/triton/discussions](https://github.com/openai/triton/discussions)
3. **PyTorch Forums:** [discuss.pytorch.org](https://discuss.pytorch.org/)

**Provide this info when asking for help:**
```bash
python -c "import torch, triton; print('Python:', __import__('sys').version); print('PyTorch:', torch.__version__); print('Triton:', triton.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```
