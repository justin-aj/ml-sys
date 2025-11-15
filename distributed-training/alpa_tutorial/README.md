# Alpa: Automated Model-Parallel Deep Learning

> **Goal:** Learn how Alpa automatically finds the best way to parallelize large models across GPUs
> 
> **Status:** ✅ Educational tutorial - Learn automated parallelism strategies
> 
> **Updated:** November 15, 2025

---

## 💡 What is Alpa?

**Alpa** is a system that **automatically** parallelizes deep learning models. Instead of manually choosing Data Parallelism, Pipeline Parallelism, or Tensor Parallelism, Alpa does it for you!

### The Problem Alpa Solves

**Manual approach (what you've been doing):**
```python
# You decide: "I'll use ZeRO-2"
CONFIG = {"strategy": "zero2"}

# You decide: "I'll use 4 pipeline stages"
pipeline = PipeDream(num_stages=4)

# You decide: "I'll shard this layer"
tensor_parallel_layer(...)
```

**Problem:** How do you know which is best? What if you combine strategies?

**Alpa's approach:**
```python
# Alpa decides automatically!
@parallelize
def train(model, data):
    return model(data)

# Alpa figures out:
# - Should we use data parallelism?
# - Should we pipeline?
# - Should we do tensor parallelism?
# - How to combine them for best speed?
```

---

## 🎯 What You'll Learn

1. **Why Automated Parallelism** - The complexity of manual parallelism
2. **Three Types of Parallelism** - Data, Pipeline, Tensor (and how Alpa combines them)
3. **Intra-op vs Inter-op Parallelism** - Alpa's two-level hierarchy
4. **Hands-on Examples** - See Alpa make automatic decisions
5. **Comparison with Manual** - ZeRO vs PipeDream vs Alpa

**Time to complete:** 2-3 hours

---

## 📚 Table of Contents

1. [The Parallelism Zoo](#the-parallelism-zoo)
2. [Alpa's Key Insight: Two-Level Hierarchy](#alpas-key-insight-two-level-hierarchy)
3. [Intra-Operator Parallelism](#intra-operator-parallelism)
4. [Inter-Operator Parallelism](#inter-operator-parallelism)
5. [How Alpa Makes Decisions](#how-alpa-makes-decisions)
6. [Hands-on Code Example](#hands-on-code-example)
7. [Comparison: Manual vs Automatic](#comparison-manual-vs-automatic)

---

## The Parallelism Zoo

There are **three main ways** to parallelize deep learning models:

### 1. Data Parallelism (What ZeRO Uses)

```
GPU0: [Full Model] → Batch 0
GPU1: [Full Model] → Batch 1
GPU2: [Full Model] → Batch 2
GPU3: [Full Model] → Batch 3
```

**Pros:** Simple, scales well for small/medium models  
**Cons:** Doesn't help if model too large for 1 GPU

### 2. Pipeline Parallelism (What PipeDream Uses)

```
GPU0: [Layers 1-12]  →
GPU1: [Layers 13-24] →
GPU2: [Layers 25-36] →
GPU3: [Layers 37-48] →
```

**Pros:** Can train very large models  
**Cons:** Pipeline bubbles waste GPU time

### 3. Tensor Parallelism (What Megatron-LM Uses)

Split individual layers across GPUs:

```
Single layer computation:
Y = X @ W  (matrix multiply)

Tensor Parallel version:
GPU0: Y0 = X @ W0  (first half of weights)
GPU1: Y1 = X @ W1  (second half of weights)
Result: Y = concat(Y0, Y1)
```

**Pros:** No pipeline bubbles, can split huge layers  
**Cons:** Lots of communication, complex to implement

---

## The Problem: Too Many Choices!

For a model with N layers and M GPUs, there are **exponentially many ways** to parallelize:

- Which layers should be pipelined?
- Which layers should be tensor-parallel?
- How many ways to split each layer?
- How to minimize communication?
- How to balance load across GPUs?

**Manual approach:** Trial and error for weeks 😰

**Alpa's approach:** Automatic optimization in minutes! 🚀

---

## Alpa's Key Insight: Two-Level Hierarchy

Alpa realizes that parallelism happens at **two levels**:

### Level 1: Intra-Operator Parallelism (Inside Operations)

**Question:** How to split a single operation (like matrix multiply) across GPUs?

```python
# Single operation
Y = X @ W  # X: [batch, 512], W: [512, 2048]

# Alpa can split this operation across 4 GPUs:
GPU0: Y0 = X @ W[:, 0:512]
GPU1: Y1 = X @ W[:, 512:1024]
GPU2: Y2 = X @ W[:, 1024:1536]
GPU3: Y3 = X @ W[:, 1536:2048]
Result: Y = concat([Y0, Y1, Y2, Y3], dim=1)
```

**This is Tensor Parallelism!**

### Level 2: Inter-Operator Parallelism (Between Operations)

**Question:** How to pipeline multiple operations across GPUs?

```python
# Multiple operations (layers)
h1 = layer1(x)
h2 = layer2(h1)
h3 = layer3(h2)
h4 = layer4(h3)

# Alpa can pipeline across 4 GPUs:
GPU0: layer1
GPU1: layer2
GPU2: layer3
GPU3: layer4
```

**This is Pipeline Parallelism!**

### Alpa Combines Both!

```
Inter-op (Pipeline): Split model into 4 stages
                     ↓
Intra-op (Tensor):   Split each stage across 2 GPUs

Result: 4 stages × 2 GPUs = 8 GPUs total
```

---

## Intra-Operator Parallelism

### What Gets Parallelized?

**Any operation can be split across GPUs:**

#### Example 1: Matrix Multiply

```python
Y = X @ W  # Shape: [batch, seq, hidden] @ [hidden, ffn]
```

**Option A: Split by columns** (Megatron-style)
```python
GPU0: Y0 = X @ W[:, 0:ffn//2]
GPU1: Y1 = X @ W[:, ffn//2:ffn]
Y = concat([Y0, Y1], dim=-1)
```

**Option B: Split by rows**
```python
GPU0: Y0 = X @ W[0:hidden//2, :]
GPU1: Y1 = X @ W[hidden//2:hidden, :]
Y = Y0 + Y1  # Reduce-sum
```

**Alpa decides which is better!**

#### Example 2: Attention

```python
# Multi-head attention has 16 heads
heads = split(Q @ K @ V, num_heads=16)

# Alpa can split heads across 4 GPUs:
GPU0: heads[0:4]
GPU1: heads[4:8]
GPU2: heads[8:12]
GPU3: heads[12:16]
```

**Alpa finds the best split automatically!**

### Communication Patterns

Different splits need different communication:

```
Column Split: All-Gather after computation
Row Split:    All-Reduce after computation
No Split:     No communication needed
```

**Alpa models communication cost** and chooses the fastest option!

---

## Inter-Operator Parallelism

### What Gets Pipelined?

**Alpa groups operators into stages:**

```python
# Model with 48 transformer layers
layer1, layer2, ..., layer48

# Alpa might decide:
Stage 0 (GPU 0): layers 1-12   (12 layers)
Stage 1 (GPU 1): layers 13-20  (8 layers)
Stage 2 (GPU 2): layers 21-35  (15 layers)
Stage 3 (GPU 3): layers 36-48  (13 layers)
```

**Notice:** Not evenly split! Alpa considers:
- Layer computation time
- Communication costs
- Memory constraints

### How Alpa Finds Stages

**Alpa uses Dynamic Programming (DP):**

1. **Build computation graph** of all operations
2. **Compute cost** of each operation (FLOPs, memory, communication)
3. **Run DP algorithm** to find best partitioning
4. **Minimize:** Total execution time (compute + communication)

**Result:** Optimal pipeline stages!

---

## How Alpa Makes Decisions

### Step 1: Profile the Model

```python
@parallelize
def train(model, batch):
    loss = model(batch)
    return loss

# Alpa traces the computation graph:
# - What operations are there?
# - How big are tensors?
# - What's the dependency order?
```

### Step 2: Generate Parallelization Options

For each operation:
```python
# Example: Matrix multiply Y = X @ W
options = [
    "no_split",           # Data parallel only
    "split_column_2",     # Split W into 2 parts (columns)
    "split_column_4",     # Split W into 4 parts
    "split_row_2",        # Split W into 2 parts (rows)
    "split_batch",        # Split X by batch dimension
]
```

### Step 3: Cost Model

For each option, estimate cost:

```python
cost = compute_time + communication_time + memory_cost

compute_time = FLOPs / GPU_speed
communication_time = bytes_to_transfer / bandwidth
memory_cost = penalty_if_exceeds_GPU_memory
```

### Step 4: Dynamic Programming (Intra-op)

Find best split for each layer:

```python
# For each layer
for layer in model.layers:
    best_option = min(options, key=lambda opt: cost(opt))
    layer.parallel_plan = best_option
```

### Step 5: Integer Linear Programming (Inter-op)

Find best pipeline stages:

```python
# Solve optimization problem:
minimize: total_execution_time
subject to:
    - Each layer assigned to exactly one stage
    - Memory per GPU ≤ GPU_capacity
    - Communication minimized
```

**Result:** Fully automated parallelization plan! 🎉

---

## Hands-on Code Example

### Installation

```bash
# Install Alpa (uses JAX, not PyTorch)
pip install alpa jax jaxlib
pip install flax  # For neural network layers

# For GPU support
pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Example 1: Simple MLP

```python
import jax
import jax.numpy as jnp
from alpa import parallelize, ShardParallel
import flax.linen as nn

# Define model (JAX/Flax style)
class MLP(nn.Module):
    hidden_size: int = 2048
    num_layers: int = 8
    
    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = nn.relu(x)
        return nn.Dense(10)(x)  # 10 classes

# Training function
@parallelize
def train_step(model, batch, labels):
    def loss_fn(params):
        logits = model.apply(params, batch)
        loss = jnp.mean((logits - labels) ** 2)
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(model.params)
    # Update params...
    return loss

# Create model
model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones([32, 512]))

# Generate fake data
batch = jnp.ones([32, 512])
labels = jnp.ones([32, 10])

# Run - Alpa automatically parallelizes!
loss = train_step(model, batch, labels)
print(f"Loss: {loss}")
```

**What Alpa does automatically:**
1. ✅ Analyzes the 8-layer MLP
2. ✅ Decides which layers to split
3. ✅ Decides how to split each matrix multiply
4. ✅ Generates efficient GPU kernels
5. ✅ Executes with optimal parallelism

**You wrote:** Just `@parallelize` decorator!

---

### Example 2: Transformer with Manual Hints

```python
from alpa import parallelize, ShardParallel
from flax import linen as nn

class TransformerBlock(nn.Module):
    hidden_size: int = 1024
    num_heads: int = 16
    
    @nn.compact
    def __call__(self, x):
        # Multi-head attention
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads
        )(x, x)
        x = x + attn_out
        x = nn.LayerNorm()(x)
        
        # Feed-forward
        ff_out = nn.Dense(self.hidden_size * 4)(x)
        ff_out = nn.relu(ff_out)
        ff_out = nn.Dense(self.hidden_size)(ff_out)
        x = x + ff_out
        x = nn.LayerNorm()(x)
        return x

class Transformer(nn.Module):
    num_layers: int = 24
    hidden_size: int = 1024
    
    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = TransformerBlock(hidden_size=self.hidden_size)(x)
        return x

# Parallelize with strategy hints
@parallelize(method=ShardParallel())
def train_step(model, batch):
    return model(batch)

# Alpa will:
# - Decide how to split attention heads
# - Decide how to split FFN layers
# - Pipeline the 24 layers
# - Minimize communication
```

---

### Example 3: See Alpa's Decisions

```python
from alpa import parallelize, get_execution_plan

@parallelize
def forward(model, x):
    return model(x)

# Run once to compile
output = forward(model, input_data)

# See what Alpa decided!
plan = get_execution_plan(forward)
print(plan)
```

**Output:**
```
Parallelization Plan:
  Total GPUs: 8
  Pipeline stages: 4
  Devices per stage: 2
  
Stage 0 (Layers 0-5):
  Device mesh: [2, 1] (2 GPUs, tensor parallel)
  Attention: split heads across 2 GPUs
  FFN: split columns across 2 GPUs
  
Stage 1 (Layers 6-11):
  Device mesh: [2, 1]
  Attention: split heads across 2 GPUs
  FFN: split columns across 2 GPUs
  
Stage 2 (Layers 12-17):
  Device mesh: [2, 1]
  ...
  
Estimated speedup: 6.3× vs single GPU
Communication volume: 2.4 GB/iteration
```

---

## Comparison: Manual vs Automatic

### Manual Parallelism (ZeRO, PipeDream)

```python
# You must decide everything:

# 1. Choose strategy
CONFIG = {"strategy": "zero2"}  # Why zero2? Why not zero3?

# 2. Choose batch size
CONFIG["batch_size"] = 4  # Will this fit? Need to try...

# 3. Choose pipeline stages
pipeline = PipeDream(num_stages=4)  # Why 4? Why not 8?

# 4. Assign layers to stages
stage0 = layers[0:12]   # Is this balanced?
stage1 = layers[12:24]
stage2 = layers[24:36]
stage3 = layers[36:48]

# 5. Choose microbatch size
num_microbatches = 4  # Optimal? Who knows...

# 6. Tune for weeks 😰
```

**Problems:**
- ❌ Requires expert knowledge
- ❌ Trial and error process
- ❌ Model-specific tuning
- ❌ No guarantees of optimality

### Automatic Parallelism (Alpa)

```python
# Alpa decides everything:

@parallelize
def train(model, batch):
    loss = model(batch)
    return loss

# That's it! 🎉
```

**Benefits:**
- ✅ No expert knowledge needed
- ✅ Automatic optimization
- ✅ Works for any model
- ✅ Near-optimal performance

---

## When to Use Alpa

### ✅ Use Alpa When:

1. **You have a complex model** with many layers and operations
2. **You want best performance** without manual tuning
3. **You're training new architectures** (Alpa adapts automatically)
4. **You have heterogeneous GPUs** (Alpa considers device differences)
5. **You want to experiment quickly** (no weeks of tuning)

### ❌ Don't Use Alpa When:

1. **Your model is small** (< 1B params) - Simple Data Parallel is fine
2. **You need PyTorch** - Alpa uses JAX only
3. **You have very custom parallelism needs** - Manual control better
4. **Production system with proven config** - No need to change

---

## Alpa vs ZeRO vs PipeDream

| Feature | ZeRO | PipeDream | Alpa |
|---------|------|-----------|------|
| **Parallelism Type** | Data + Memory | Pipeline | All 3 types |
| **Automatic?** | Semi (manual config) | No (manual stages) | ✅ Fully automatic |
| **Framework** | PyTorch | PyTorch | JAX only |
| **Best For** | Medium models (1-10B) | Very deep models | Any large model |
| **Optimization** | Memory focused | Pipeline focused | Speed + memory |
| **Ease of Use** | Medium | Hard | ✅ Easy |
| **Performance** | Fast | Medium (bubbles) | ✅ Fastest |

**Summary:** Alpa combines the benefits of all approaches automatically!

---

## Real-World Example: GPT-3

Imagine training GPT-3 (175B params) on 64 GPUs:

### Manual Approach (Megatron-LM)

```python
# Expert team spends weeks deciding:
- Pipeline parallelism: 8 stages (8 GPUs per stage)
- Tensor parallelism: 8-way split within each stage
- Data parallelism: 1 (no data parallelism)
- Microbatch size: 4
- Layer assignment: carefully balanced
Total time: 3 weeks of tuning
```

### Alpa Approach

```python
@parallelize
def train_gpt3(model, batch):
    return model(batch)

# Alpa figures it out in 20 minutes:
- Found optimal pipeline: 8 stages
- Found optimal tensor splits: 8-way for FFN, 4-way for attention
- Auto-batching: determined optimal microbatch
- Result: 5% faster than manual!
Total time: 20 minutes
```

---

## Advanced: How Alpa's Algorithms Work

### Intra-Operator: Dynamic Programming

```python
# For each operation (e.g., matrix multiply)
def find_best_sharding(op):
    # All possible ways to split
    options = generate_sharding_options(op)
    
    # Cost model
    costs = [compute_cost(opt) + comm_cost(opt) for opt in options]
    
    # Choose minimum
    return options[argmin(costs)]
```

**Time complexity:** O(N × S²) where N = operations, S = sharding options

### Inter-Operator: Integer Linear Programming (ILP)

```python
# Optimization problem
minimize: T_total
subject to:
    sum(assign[i,s]) == 1  for all operations i  # Each op in one stage
    sum(memory[i] * assign[i,s]) <= GPU_mem  for all stages s
    T_total >= max(stage_time[s])  for all s
    
variables:
    assign[i,s] ∈ {0,1}  # Is operation i in stage s?
    T_total ≥ 0
```

**Solved with:** Off-the-shelf ILP solver (Gurobi or CPLEX)

---

## Key Innovations in Alpa

### 1. Two-Level Hierarchy

**Problem:** Joint optimization of intra-op and inter-op is NP-hard

**Solution:** Decouple into two levels:
- Level 1: Optimize intra-op (tensor parallelism) for each layer
- Level 2: Optimize inter-op (pipeline) given level 1 results

**Result:** Near-optimal in polynomial time!

### 2. Device Mesh Abstraction

Alpa thinks of GPUs as a 2D mesh:

```
8 GPUs arranged as 4×2 mesh:

Stage 0: [GPU0, GPU1]  ← 2 GPUs tensor-parallel
Stage 1: [GPU2, GPU3]
Stage 2: [GPU4, GPU5]
Stage 3: [GPU6, GPU7]

4 stages pipeline-parallel
```

**Benefit:** Can optimize communication topology!

### 3. Compilation and Code Generation

Alpa generates optimized GPU kernels:

```python
# High-level: Y = X @ W
# Alpa generates:
GPU0: Y0 = AllGather(X) @ W0; AllReduce(Y0)
GPU1: Y1 = AllGather(X) @ W1; AllReduce(Y1)
```

**Result:** Communication overlapped with computation!

---

## Hands-On Exercise

Try Alpa yourself:

```bash
cd alpa_tutorial
pip install -r requirements.txt
python alpa_simple.py
```

**What you'll see:**
1. Model definition (simple transformer)
2. Alpa's automatic parallelization
3. Execution plan (what Alpa decided)
4. Performance comparison (automatic vs manual)

---

## Summary

### What You Learned

✅ **Three types of parallelism**: Data, Pipeline, Tensor  
✅ **Alpa's two levels**: Intra-op and Inter-op  
✅ **Automatic optimization**: No manual tuning needed  
✅ **Cost modeling**: How Alpa estimates performance  
✅ **Dynamic Programming**: How Alpa finds optimal intra-op sharding  
✅ **Integer Linear Programming**: How Alpa finds optimal inter-op pipeline  
✅ **Hands-on examples**: JAX/Flax code with `@parallelize`

### Key Takeaways

1. **Manual parallelism is hard** - Too many choices, requires expertise
2. **Alpa automates everything** - Just add `@parallelize` decorator
3. **Two-level hierarchy** - Decomposes hard problem into tractable subproblems
4. **Near-optimal performance** - Often matches or beats manual tuning
5. **JAX-only** - Works with JAX/Flax, not PyTorch (for now)

---

## Next Steps

1. **Try the examples** - Run `alpa_simple.py`
2. **Read the paper** - [Alpa: Automating Inter- and Intra-Operator Parallelism](https://arxiv.org/abs/2201.12023)
3. **Compare with ZeRO** - See `../` for ZeRO tutorial
4. **Compare with PipeDream** - See `../pipedream_tutorial/`
5. **Experiment** - Try Alpa on your own models!

---

## Additional Resources

- **Alpa Paper**: https://arxiv.org/abs/2201.12023
- **Alpa GitHub**: https://github.com/alpa-projects/alpa
- **Google Blog Post**: https://research.google/blog/alpa-automated-model-parallel-deep-learning/
- **JAX Tutorial**: https://jax.readthedocs.io/
- **Flax Tutorial**: https://flax.readthedocs.io/

---

**Ready to learn automated parallelism? Let's go! 🚀**
