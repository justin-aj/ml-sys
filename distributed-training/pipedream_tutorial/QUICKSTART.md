# PipeDream Quick Start Guide

⚡ **Get started in 5 minutes!**

⚠️ **Only have 1 GPU?** Perfect! This tutorial **simulates** pipeline parallelism on a single GPU for learning.

---

## 💡 Important: Single GPU Simulation

**This tutorial works on 1 GPU (or even CPU)!**

- ✅ Simulates what would happen with 4 GPUs
- ✅ Teaches all PipeDream concepts
- ✅ No multi-GPU setup needed
- ✅ Perfect for learning before scaling to real clusters

**When you get multi-GPU access:** The concepts transfer directly to real PipeDream implementations!

---

## Step 1: Install Dependencies

```bash
pip install torch matplotlib numpy
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

**Note:** Works on CPU too! No GPU required for learning.

---

## Step 2: Run the Simple Demo

```bash
python pipedream_simple.py
```

**What you'll see:**
- Pipeline creation with 4 stages (GPUs)
- Timeline visualization showing forward/backward passes
- Microbatch processing demonstration
- Weight versioning in action

**Output example:**
```
🚀 PipeDream Pipeline Created!
⚠️  SIMULATION MODE: Running on 1 GPU (simulating 4 GPUs)
Pipeline stages: 4 (would be 4 GPUs in real setup)
Microbatches: 4

💡 In real PipeDream:
   - Each stage would be on a separate GPU
   - All 4 GPUs would work in parallel

💡 In this simulation:
   - All stages are on 1 GPU (for learning)
   - Timeline shows what WOULD happen on 4 GPUs

📊 PIPELINE TIMELINE VISUALIZATION
Time | GPU0 (Stage 0) | GPU1 (Stage 1) | GPU2 (Stage 2) | GPU3 (Stage 3)
t3   | MB3 forward    | MB2 forward    | MB1 forward    | MB0 forward    ← All busy!

BATCH 1/5
Average loss across 4 microbatches: 2.3327
Stage 0: Updated weights v0 → v1

✅ Batch complete! All stages now at version 1
```

**Training results (5 batches):**
- Batch 1: Loss 2.3327 (weights v0 → v1)
- Batch 2: Loss 2.3061 (weights v1 → v2)
- Batch 3: Loss 2.3256 (weights v2 → v3)
- Batch 4: Loss 2.3003 (weights v3 → v4)
- Batch 5: Loss 2.3312 (weights v4 → v5)

**See TEST_RESULTS.md for complete analysis!**

**Don't worry about the "simulation" messages - that's the point! You're learning the concepts.**

---

## Step 3: Generate Visualizations

```bash
python pipedream_visual.py
```

**What you'll get:**
- `naive_pipeline.png` - Shows why naive pipeline is inefficient (only 1 GPU working)
- `microbatch_forward.png` - Forward pass timeline (3-4 GPUs working!)
- `microbatch_backward.png` - Backward pass timeline (gradients in reverse)
- `weight_versioning.png` - Weight versioning concept (all MBs use same W₀)
- `utilization_comparison.png` - Efficiency comparison (25% vs 75% vs 90%)

**Visual proof:**

These images show exactly what the simulation demonstrates:

1. **Naive Pipeline** - Only 1 GPU working = 25% utilization ❌
2. **Microbatch Forward** - 3-4 GPUs working = 75% utilization ✅
3. **Microbatch Backward** - All GPUs busy with gradients ✅
4. **Weight Versioning** - All microbatches use W₀, then update to W₁ ✅
5. **Utilization Chart** - Clear improvement: 25% → 75% → 90% ✅

**Open the PNG files to see the diagrams!**

---

## Step 4: Experiment with CONFIG

Edit `pipedream_simple.py` and change CONFIG values:

```python
CONFIG = {
    "num_stages": 4,          # Try: 2, 4, 8
    "num_microbatches": 4,    # Try: 2, 4, 8
    "layers_per_stage": 3,    # Layers per GPU
    "verbose": True,          # Set False for less output
}
```

**Experiments to try:**
1. **Fewer microbatches (2)** - See more idle time
2. **More microbatches (8)** - See better utilization
3. **More stages (8)** - Simulate larger models

---

## Understanding the Output

### Timeline Visualization

The script shows when each GPU is working:

```
Time | GPU0 (Stage 0) | GPU1 (Stage 1) | GPU2 (Stage 2) | GPU3 (Stage 3)
t0   | MB0 forward    | idle          | idle          | idle
t1   | MB1 forward    | MB0 forward   | idle          | idle
t2   | MB2 forward    | MB1 forward   | MB0 forward   | idle
t3   | MB3 forward    | MB2 forward   | MB1 forward   | MB0 forward
```

**Notice:** At t3, all 4 GPUs are working! This is the power of microbatches.

### Weight Versioning

```
MB0 forward  → uses W0
MB1 forward  → uses W0 (same version!)
MB2 forward  → uses W0
MB3 forward  → uses W0

All backward passes compute gradients for W0
Then: W1 = W0 - learning_rate * (all gradients)
```

**Key point:** All microbatches use the SAME weight version for consistency.

---

## Common Questions

**Q: I only have 1 GPU, will this work?**
A: ✅ YES! This tutorial **simulates** multi-GPU pipeline parallelism on 1 GPU. Perfect for learning!

**Q: Do I need multiple GPUs to run this?**
A: ❌ NO! The tutorial simulates 4 GPUs on your single GPU (or even CPU). You'll learn all the concepts.

**Q: Will this teach me real PipeDream?**
A: ✅ YES! All concepts (microbatches, weight versioning, pipeline stages) are the same. When you get multi-GPU access, you'll understand how to use real PipeDream.

**Q: Why split into microbatches?**
A: To keep all GPUs busy! Without microbatches, only 1 GPU works at a time (25% utilization).

**Q: What is weight versioning?**
A: Ensuring all microbatches use the same weights during forward pass, so gradients are consistent.

**Q: When to use pipeline parallelism in production?**
A: When your model is too large to fit on one GPU, but fits when split across multiple GPUs.

**Q: PipeDream vs Data Parallelism?**
A: 
- **Data Parallelism**: Same model on each GPU, different data
- **Pipeline Parallelism**: Different parts of model on each GPU, same data flows through

---

## Next Steps

1. ✅ **Read README.md** - Full tutorial with detailed explanations
2. ✅ **Run pipedream_simple.py** - See the code in action
3. ✅ **Run pipedream_visual.py** - Generate visual diagrams
4. ✅ **Experiment with CONFIG** - Try different settings
5. ✅ **Compare with ZeRO** - See `../` for ZeRO tutorial

---

## Troubleshooting

**Issue: "ModuleNotFoundError: No module named 'torch'"**
```bash
pip install torch
```

**Issue: "No display found" (for visualizations)**
```bash
# Add this at the top of pipedream_visual.py:
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

**Issue: Want to run without GPU**
- ✅ This tutorial works on CPU! No GPU needed for the demo.
- The simulation runs fine on CPU - it's for learning, not performance.

**Issue: "CUDA out of memory"**
- ✅ Reduce batch_size in CONFIG (try 16 or 8)
- Or run on CPU: `device = torch.device('cpu')` in code

---

## Learning Path

**Level 1 (Beginner):** 
- Read README.md sections 1-3
- Run `pipedream_simple.py` with verbose=True
- Look at generated visualizations

**Level 2 (Intermediate):**
- Read README.md sections 4-5
- Experiment with different CONFIG settings
- Compare timeline outputs

**Level 3 (Advanced):**
- Read PipeDream paper (linked in README.md)
- Modify code to add real training loop
- Compare with GPipe, Megatron-LM

---

**Ready? Start here:**
```bash
python pipedream_simple.py
```

🚀 **Happy learning!**
