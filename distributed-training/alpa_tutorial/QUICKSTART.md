# Alpa Quick Start - 10 Minutes

⚡ **Learn automated model parallelism in 10 minutes!**

---

## What is Alpa?

**Alpa automatically decides** how to parallelize your model across multiple GPUs.

**You:** Just add `@parallelize` decorator  
**Alpa:** Figures out the best way to split your model

---

## Installation

```bash
# Install Alpa and JAX
pip install alpa
pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax optax

# Or use requirements.txt
pip install -r requirements.txt
```

**Note:** Alpa uses JAX (not PyTorch). JAX is like NumPy + automatic differentiation.

---

## Your First Alpa Program

```python
import jax
import jax.numpy as jnp
from alpa import parallelize
import flax.linen as nn

# 1. Define model (JAX/Flax style)
class SimpleMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(2048)(x)
        x = nn.relu(x)
        x = nn.Dense(2048)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

# 2. Add @parallelize decorator
@parallelize
def train_step(model, batch):
    logits = model(batch)
    loss = jnp.mean(logits ** 2)
    return loss

# 3. Run - Alpa parallelizes automatically!
model = SimpleMLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones([32, 512]))

batch = jnp.ones([32, 512])
loss = train_step.apply(params, batch)

print(f"Loss: {loss}")
print("✅ Alpa automatically parallelized your model!")
```

**That's it!** Alpa decided how to split the model across GPUs.

---

## What Alpa Does Automatically

When you run the code above, Alpa:

1. ✅ **Analyzes your model** - Finds all operations (Dense, ReLU, etc.)
2. ✅ **Generates parallelization options** - How to split each layer
3. ✅ **Estimates costs** - Compute time + communication time
4. ✅ **Finds optimal plan** - Using dynamic programming
5. ✅ **Generates GPU code** - Efficient kernels for your parallelization
6. ✅ **Executes** - Runs on all GPUs in parallel

**You wrote:** 5 lines (define model + `@parallelize`)  
**Manual approach would need:** 100+ lines of parallelization code!

---

## Example: See Alpa's Decisions

```python
from alpa import parallelize, get_last_executable

@parallelize
def forward(params, x):
    return model.apply(params, x)

# Run once
output = forward(params, input_data)

# See what Alpa decided
executable = get_last_executable(forward)
print(executable.get_hlo_text())  # See the parallelization plan
```

---

## Three Types of Parallelism Alpa Uses

### 1. Data Parallelism
```
GPU0: [Full Model] → Batch 0
GPU1: [Full Model] → Batch 1
```
**When:** Model fits on one GPU

### 2. Tensor Parallelism
```
Single layer: Y = X @ W
GPU0: Y0 = X @ W[:, 0:half]
GPU1: Y1 = X @ W[:, half:end]
```
**When:** Individual layers are huge

### 3. Pipeline Parallelism
```
GPU0: Layers 1-8
GPU1: Layers 9-16
GPU2: Layers 17-24
```
**When:** Model has many layers

**Alpa combines all three automatically!**

---

## Comparison with Manual Approaches

### Manual (ZeRO, PipeDream)
```python
# You must decide:
CONFIG = {
    "strategy": "zero2",      # Why zero2? Why not zero3?
    "pipeline_stages": 4,     # Why 4? Optimal?
    "tensor_parallel": 2,     # Should this be 4?
    "microbatches": 8,        # Is this the best?
}

# Then write 100+ lines to implement it...
```

### Alpa (Automatic)
```python
@parallelize
def train(model, batch):
    return loss

# Alpa decides everything! 🎉
```

---

## When to Use Alpa

✅ **Use Alpa if:**
- You want best performance without manual tuning
- You're trying new model architectures
- You don't have time to tune parallelization
- You have complex models (transformers, mixture-of-experts)

❌ **Don't use Alpa if:**
- Your model is tiny (< 100M params)
- You must use PyTorch (Alpa is JAX-only)
- You have a proven config that works

---

## Run the Examples

```bash
# Simple MLP example
python alpa_simple.py

# See parallelization plan
python alpa_visualize.py
```

**What you'll see:**
- Automatic parallelization in action
- Alpa's decisions for each layer
- Performance comparison

---

## Key Concepts (5-Minute Read)

### Intra-Operator Parallelism
**Splitting individual operations across GPUs**

```python
# Matrix multiply: Y = X @ W
# Alpa can split W across 4 GPUs:
GPU0: Y0 = X @ W[:, 0:1/4]
GPU1: Y1 = X @ W[:, 1/4:1/2]
GPU2: Y2 = X @ W[:, 1/2:3/4]
GPU3: Y3 = X @ W[:, 3/4:end]
Result: Y = concat([Y0, Y1, Y2, Y3])
```

### Inter-Operator Parallelism
**Pipelining different operations across GPUs**

```python
# 4 layers
GPU0: layer1 → 
GPU1: layer2 → 
GPU2: layer3 → 
GPU3: layer4 →
```

### Alpa's Two-Level Optimization

1. **Level 1:** Find best intra-op split for each operation (dynamic programming)
2. **Level 2:** Find best inter-op pipeline (integer linear programming)

**Result:** Near-optimal parallelization automatically!

---

## Common Questions

**Q: I only have 1 GPU, will this work?**  
A: Yes! Alpa will use data parallelism (no splitting). Good for learning.

**Q: Does Alpa work with PyTorch?**  
A: No, Alpa uses JAX. But JAX is easy to learn (similar to NumPy/PyTorch).

**Q: Is Alpa better than manual tuning?**  
A: Often yes! Alpa finds configurations experts miss. Sometimes 10-20% faster.

**Q: How long does Alpa take to optimize?**  
A: Usually 5-30 minutes for compilation, then runs fast.

**Q: Can I give Alpa hints?**  
A: Yes! You can specify constraints (e.g., "use at most 4-way tensor parallel").

---

## Next Steps

1. ✅ Read `README.md` - Full tutorial with details
2. ✅ Run `alpa_simple.py` - See it in action
3. ✅ Try your own model - Just add `@parallelize`!
4. ✅ Read the paper - [Alpa: Automating Inter- and Intra-Operator Parallelism](https://arxiv.org/abs/2201.12023)
5. ✅ Compare with ZeRO and PipeDream tutorials

---

## Troubleshooting

**Issue: "No module named 'jax'"**
```bash
pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Issue: "No CUDA-capable device is detected"**
- Alpa works on CPU too! Slower but good for learning.

**Issue: "Compilation takes forever"**
- First run compiles (5-30 min). Subsequent runs are fast.
- Use smaller model for testing.

---

**Ready? Start here:**
```bash
python alpa_simple.py
```

🚀 **Learn automatic parallelism in minutes!**
