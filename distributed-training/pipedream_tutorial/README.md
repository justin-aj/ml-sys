# PipeDream: Pipeline Parallelism Tutorial

> **Goal:** Understand how PipeDream keeps all GPUs busy using microbatches and weight versioning
> 
> **Status:** ✅ Educational tutorial - Works on **1 GPU** (simulates multi-GPU behavior)
> 
> **Updated:** November 15, 2025

---

## 💡 Important: Single GPU Simulation

**You only have 1 GPU?** No problem! This tutorial **simulates** pipeline parallelism concepts on a single GPU.

- ✅ Learn all PipeDream concepts without needing multiple GPUs
- ✅ Code shows how microbatches and weight versioning work
- ✅ Timeline visualizations show what would happen on 4 GPUs
- ✅ When you get multi-GPU access, the concepts transfer directly!

**Note:** Real PipeDream needs multiple GPUs, but this tutorial teaches the core ideas.

---

## 🎯 What You'll Learn

1. **Pipeline Parallelism Basics** - How to split models across GPUs
2. **Microbatches** - Why they keep GPUs busy instead of idle
3. **Weight Versioning** - PipeDream's key innovation
4. **Hands-on Implementation** - Working code that simulates the concepts

**Time to complete:** 1-2 hours

---

## 📚 Table of Contents

1. [Why Pipeline Parallelism?](#why-pipeline-parallelism)
2. [The Idle GPU Problem](#the-idle-gpu-problem)
3. [Microbatches to the Rescue](#microbatches-to-the-rescue)
4. [PipeDream's Innovation: Weight Versioning](#pipedreams-innovation-weight-versioning)
5. [Hands-on Code Example](#hands-on-code-example)
6. [Visual Timeline](#visual-timeline)

---

## Why Pipeline Parallelism?

### The Problem: Model Too Large for One GPU

Imagine you have a huge neural network with 48 layers, but each GPU can only fit 12 layers:

```
Model: [L1 L2 L3 ... L48]  ❌ Too big for 1 GPU!

Solution: Split across 4 GPUs
GPU0: [L1  → L12]
GPU1: [L13 → L24]
GPU2: [L25 → L36]
GPU3: [L37 → L48]
```

**This is Pipeline Parallelism!** Each GPU gets a "stage" of the model.

---

## The Idle GPU Problem

### Naive Pipeline: Lots of Wasted Time

If we send just one batch through:

```
Time →
t0:  GPU0[Forward] → GPU1[idle] → GPU2[idle] → GPU3[idle]
t1:  GPU0[idle] → GPU1[Forward] → GPU2[idle] → GPU3[idle]
t2:  GPU0[idle] → GPU1[idle] → GPU2[Forward] → GPU3[idle]
t3:  GPU0[idle] → GPU1[idle] → GPU2[idle] → GPU3[Forward]
```

❌ **Only 1 GPU working at a time!** The others are idle = wasted money!

**GPU Utilization:** ~25% (terrible!)

---

## Microbatches to the Rescue

### Solution: Split Batch into Microbatches

Instead of sending 1 big batch, split into 4 microbatches (MB0, MB1, MB2, MB3):

### Forward Pass Timeline

| Time | GPU0 (L1-12) | GPU1 (L13-24) | GPU2 (L25-36) | GPU3 (L37-48) |
|------|--------------|---------------|---------------|---------------|
| t0   | MB0 →        | idle          | idle          | idle          |
| t1   | MB1 →        | MB0 →         | idle          | idle          |
| t2   | MB2 →        | MB1 →         | MB0 →         | idle          |
| t3   | MB3 →        | MB2 →         | MB1 →         | MB0 →         |
| t4   | idle         | MB3 →         | MB2 →         | MB1 →         |
| t5   | idle         | idle          | MB3 →         | MB2 →         |
| t6   | idle         | idle          | idle          | MB3 →         |

✅ **Now 3-4 GPUs working simultaneously!** Much better!

### Backward Pass Timeline

Gradients flow **backward** (GPU3 → GPU0):

| Time | GPU0 (L1-12) | GPU1 (L13-24) | GPU2 (L25-36) | GPU3 (L37-48) |
|------|--------------|---------------|---------------|---------------|
| t7   | idle         | idle          | idle          | ← MB0 grad    |
| t8   | idle         | idle          | ← MB0 grad    | ← MB1 grad    |
| t9   | idle         | ← MB0 grad    | ← MB1 grad    | ← MB2 grad    |
| t10  | ← MB0 grad   | ← MB1 grad    | ← MB2 grad    | ← MB3 grad    |
| t11  | ← MB1 grad   | ← MB2 grad    | ← MB3 grad    | idle          |
| t12  | ← MB2 grad   | ← MB3 grad    | idle          | idle          |
| t13  | ← MB3 grad   | idle          | idle          | idle          |

**GPU Utilization:** ~75% (much better!)

---

## PipeDream's Innovation: Weight Versioning

### The Problem with Naive Microbatches

Look at time t10 above:
- GPU0 is doing **backward for MB0** (needs to update weights)
- But GPU0 is also doing **forward for MB3** at other times!

**Question:** Which weights should MB3's forward pass use?
- The **original weights W0**? (before any updates)
- Or **updated weights W1**? (after MB0's gradients applied)

❌ **If we update weights immediately, later microbatches see inconsistent weights!**

### PipeDream's Solution: Weight Versioning

**Key Idea:** Each microbatch "locks in" the weight version it uses:

```
MB0 forward  → uses W0
MB0 backward → creates gradients for W0 → updates to W1

MB1 forward  → uses W0 (same as MB0!)
MB1 backward → creates gradients for W0 → updates to W1

...all microbatches use W0...

After all microbatches backward → apply all gradients → W1
```

**In PipeDream:**
1. All microbatches in the **same batch** use the **same weight version**
2. Weights update **only after all microbatches finish**
3. This ensures **correctness** - gradients correspond to the weights used in forward

### Visual Example

```
Forward passes (all use W0):
MB0: GPU0[W0] → GPU1[W0] → GPU2[W0] → GPU3[W0]
MB1: GPU0[W0] → GPU1[W0] → GPU2[W0] → GPU3[W0]
MB2: GPU0[W0] → GPU1[W0] → GPU2[W0] → GPU3[W0]
MB3: GPU0[W0] → GPU1[W0] → GPU2[W0] → GPU3[W0]

Backward passes (all compute grads for W0):
MB0: GPU0[grad→W0] ← GPU1[grad→W0] ← GPU2[grad→W0] ← GPU3[grad→W0]
MB1: GPU0[grad→W0] ← GPU1[grad→W0] ← GPU2[grad→W0] ← GPU3[grad→W0]
MB2: GPU0[grad→W0] ← GPU1[grad→W0] ← GPU2[grad→W0] ← GPU3[grad→W0]
MB3: GPU0[grad→W0] ← GPU1[grad→W0] ← GPU2[grad→W0] ← GPU3[grad→W0]

Weight update:
W1 = W0 - learning_rate * (grad_MB0 + grad_MB1 + grad_MB2 + grad_MB3)

Next batch uses W1, and the cycle repeats!
```

---

## Key Insights

### 1. Microbatches ≠ Forward Pass Only
- Microbatches keep the pipeline busy during **both** forward and backward
- Each GPU processes microbatches one at a time

### 2. Weight Versioning = Correctness
- Ensures backward pass updates the **correct** weight version
- Without it, gradients would be inconsistent!

### 3. Pipeline Efficiency
- **Naive pipeline:** 25% GPU utilization (1 GPU working)
- **Microbatch pipeline:** 75% GPU utilization (3-4 GPUs working)
- **Trade-off:** Extra memory for weight versions

---

## PipeDream vs Plain Pipeline Parallelism

| Feature | Plain Pipeline | PipeDream |
|---------|----------------|-----------|
| **Microbatches** | Optional | Required |
| **GPU Utilization** | Low (~25%) | High (~75%) |
| **Weight Versioning** | No | Yes |
| **Memory per GPU** | Lower | Higher (stores weight versions) |
| **Complexity** | Simple | Moderate (version tracking) |
| **Best for** | Small models | Large models that don't fit on 1 GPU |

---

## Hands-on Code Example

See `pipedream_simple.py` for a working implementation that demonstrates:
1. Splitting a model across 4 GPUs
2. Processing microbatches in a staggered manner
3. Weight versioning to ensure correctness

**Key CONFIG options:**
```python
CONFIG = {
    "num_gpus": 4,           # Pipeline stages
    "num_microbatches": 4,   # Split batch into 4
    "model_layers": 48,      # Total layers
    "batch_size": 32,        # Total batch size
}
```

---

## 🚀 Running the Tutorial

### Installation

```bash
# Install dependencies
pip install torch matplotlib numpy

# Or use requirements.txt
pip install -r requirements.txt
```

### Run the Educational Simulation

```bash
python pipedream_simple.py
```

**What you'll see:**
- ⚠️ Simulation mode messages (running on 1 GPU)
- Timeline visualization showing GPU activity
- 5 batches trained with microbatch scheduling
- Weight versioning demonstrated (v0 → v1 → v2 → v3 → v4 → v5)
- All microbatches using same weight version per batch

**Time to run:** ~30 seconds (simulation only, not real training)

### Generate Visual Diagrams

```bash
python pipedream_visual.py
```

**Creates 5 PNG images:**
1. `naive_pipeline.png` - Shows idle GPU problem (25% utilization)
2. `microbatch_forward.png` - Forward pass timeline (3-4 GPUs working!)
3. `microbatch_backward.png` - Backward pass timeline (gradients in reverse)
4. `weight_versioning.png` - Weight versioning concept (all MBs use W₀)
5. `utilization_comparison.png` - Efficiency comparison (25% → 75% → 90%)

**Generated images:**

![Naive Pipeline](naive_pipeline.png)
*Naive pipeline: Only 1 GPU working at a time = 25% utilization (terrible!)*

![Microbatch Forward](microbatch_forward.png)
*Microbatch pipeline forward pass: 3-4 GPUs working simultaneously = 75% utilization!*

![Microbatch Backward](microbatch_backward.png)
*Microbatch pipeline backward pass: Gradients flow in reverse, all GPUs busy!*

![Weight Versioning](weight_versioning.png)
*PipeDream weight versioning: All microbatches use W₀, creating W₁*

![Utilization Comparison](utilization_comparison.png)
*GPU utilization comparison: 25% → 75% → 90% (3× better!)*

---

## 📊 Actual Test Results

**Tested:** November 15, 2025 (Single GPU - Simulation Mode)  
**Configuration:** 4 stages, 4 microbatches, batch size 32

### Sample Output

```
🚀 PipeDream Pipeline Created!
⚠️  SIMULATION MODE: Running on 1 GPU (simulating 4 GPUs)
Pipeline stages: 4 (would be 4 GPUs in real setup)
Microbatches: 4

📊 PIPELINE TIMELINE VISUALIZATION
Forward Pass Timeline:
Time | GPU0 (Stage 0) | GPU1 (Stage 1) | GPU2 (Stage 2) | GPU3 (Stage 3)
t0   | MB0 forward    | idle          | idle          | idle
t1   | MB1 forward    | MB0 forward   | idle          | idle
t2   | MB2 forward    | MB1 forward   | MB0 forward   | idle
t3   | MB3 forward    | MB2 forward   | MB1 forward   | MB0 forward   ← All busy!

BATCH 1/5
📤 Forward pass for microbatch 0
  Stage 0: MB0 → using weights v0 → output shape torch.Size([8, 512])
  Stage 1: MB0 → using weights v0 → output shape torch.Size([8, 512])
  Stage 2: MB0 → using weights v0 → output shape torch.Size([8, 512])
  Stage 3: MB0 → using weights v0 → output shape torch.Size([8, 10])

WEIGHT UPDATE PHASE
Average loss across 4 microbatches: 2.3327
Stage 0: Updated weights v0 → v1
Stage 1: Updated weights v0 → v1
Stage 2: Updated weights v0 → v1
Stage 3: Updated weights v0 → v1

✅ Batch complete! All stages now at version 1
```

### Training Results (5 Batches)

| Batch | Loss | Weight Version Used | Weight Version After |
|-------|------|-------------------|---------------------|
| 1 | 2.3327 | v0 | v1 |
| 2 | 2.3061 | v1 | v2 |
| 3 | 2.3256 | v2 | v3 |
| 4 | 2.3003 | v3 | v4 |
| 5 | 2.3312 | v4 | v5 |

**Key Observations:**
- ✅ All 4 microbatches per batch use the **same weight version**
- ✅ Weights update only **after all microbatches** complete
- ✅ Timeline clearly shows **3-4 GPUs working simultaneously**
- ✅ Forward and backward phases are **staggered** across stages

---

## What's Next?

1. **Understand the basics** - Read this README carefully
2. **Run the code** - See microbatches and weight versioning in action
3. **Visualize the timeline** - Use the visualization script
4. **Compare with ZeRO** - Pipeline parallelism vs model parallelism

---

## Additional Resources

- **PipeDream Paper:** "PipeDream: Generalized Pipeline Parallelism for DNN Training" (2019)
- **GPipe Paper:** "GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism" (2019)
- **PyTorch Pipeline Docs:** https://pytorch.org/docs/stable/pipeline.html

---

## Summary

**Pipeline Parallelism:** Split model layers across GPUs

**Problem:** GPUs idle most of the time

**Solution:** Microbatches keep all GPUs busy

**PipeDream:** Weight versioning ensures correctness

**Result:** 3× better GPU utilization! 🚀

---

**Next:** Open `pipedream_simple.py` to see the code!
