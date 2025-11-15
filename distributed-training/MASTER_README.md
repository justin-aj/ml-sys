# ML Systems: Distributed Training Tutorials

> **Complete guide to distributed training strategies with real models and hands-on examples**
> 
> **Updated:** November 15, 2025 | **Status:** ✅ Tested on GPU cluster

---

## 📚 What's Inside

This repository contains **two comprehensive tutorials**:

### 1. **ZeRO Tutorial** (`strategies/` + root files)
Learn DeepSpeed ZeRO for memory-efficient distributed training
- 6 strategies from Data Parallelism to ZeRO-Infinity
- Real GPT-2 models (124M to 1.5B parameters)
- Tested results with performance comparisons
- Works on 1 or multiple GPUs

### 2. **PipeDream Tutorial** (`pipedream_tutorial/`)
Learn pipeline parallelism with microbatches and weight versioning
- Educational simulation (works on 1 GPU!)
- Visual diagrams and timeline explanations
- Hands-on code demonstrating key concepts
- Perfect for understanding before scaling

---

## 🚀 Quick Start

### Option 1: ZeRO Tutorial (Recommended First)

```bash
# Install dependencies
pip install -r requirements.txt

# Run with ZeRO-2 (best performance)
cd strategies/3_zero_stage2
bash run.sh
```

**What you'll learn:**
- How ZeRO shards optimizer states, gradients, and parameters
- Memory savings and speed improvements
- When to use each ZeRO stage

**Time:** 30-60 minutes

### Option 2: PipeDream Tutorial

```bash
# Navigate to tutorial
cd pipedream_tutorial

# Install dependencies
pip install -r requirements.txt

# Run simulation
python pipedream_simple.py

# Generate visual diagrams
python pipedream_visual.py
```

**What you'll learn:**
- How pipeline parallelism splits models across GPUs
- Why microbatches improve GPU utilization
- How weight versioning ensures correctness

**Time:** 1-2 hours

---

## 📊 Strategy Comparison

| Strategy | Memory Savings | Speed | Complexity | Best For |
|----------|---------------|-------|------------|----------|
| **Data Parallel** | Baseline | 1.0× | Simple | Small models |
| **ZeRO-1** | 4× | 2.4× | Easy | Optimizer-heavy |
| **ZeRO-2** | 8× | 2.4× | Easy | **Recommended** ✅ |
| **ZeRO-3** | 64× | 0.8× | Medium | Large models |
| **ZeRO-Offload** | 52% GPU | 0.3× | Medium | Limited GPU memory |
| **ZeRO-Infinity** | Unlimited | Varies | Advanced | Massive models |
| **Pipeline Parallel** | Model split | 0.75× | Advanced | Very deep models |

**Speedup** = relative to Data Parallel baseline

---

## 📁 Repository Structure

```
distributed-training/
│
├── strategies/                      # ZeRO strategies (6 approaches)
│   ├── README.md                   # Overview of all strategies
│   ├── 1_data_parallel/            # Baseline
│   ├── 2_zero_stage1/              # Optimizer sharding
│   ├── 3_zero_stage2/              # Optimizer + gradient sharding ✅
│   ├── 4_zero_stage3/              # Full parameter sharding
│   ├── 5_zero_offload/             # CPU memory offloading
│   └── 6_zero_infinity/            # NVMe storage
│
├── pipedream_tutorial/              # Pipeline parallelism
│   ├── README.md                   # Main tutorial
│   ├── QUICKSTART.md               # 5-minute guide
│   ├── TEST_RESULTS.md             # Actual test results
│   ├── COMPARISON.md               # PipeDream vs ZeRO
│   ├── pipedream_simple.py         # Educational simulation
│   ├── pipedream_visual.py         # Generate diagrams
│   ├── *.png                       # 5 visualization images
│   └── requirements.txt
│
├── real_model_example.py            # Main training script (works with all strategies)
├── README.md                        # ZeRO tutorial main doc
├── TRAINING_RESULTS.md              # ZeRO test results
├── REAL_MODELS_GUIDE.md             # Advanced GPT-2 guide
├── CLUSTER_QUICKSTART.md            # Multi-node setup
├── MASTER_README.md                 # This file
└── requirements.txt                 # Dependencies
```

---

## 🎯 Learning Path

### Beginner (1-2 hours)
1. Read `README.md` (ZeRO tutorial intro)
2. Run Data Parallelism baseline
3. Run ZeRO-2 and compare results
4. Read `pipedream_tutorial/QUICKSTART.md`
5. Run `pipedream_simple.py`

### Intermediate (3-5 hours)
1. Read `TRAINING_RESULTS.md` (performance analysis)
2. Test all 6 ZeRO strategies
3. Read `pipedream_tutorial/README.md` (full tutorial)
4. Run `pipedream_visual.py` and study diagrams
5. Read `COMPARISON.md` (when to use what)

### Advanced (5+ hours)
1. Read `REAL_MODELS_GUIDE.md` (advanced techniques)
2. Read `CLUSTER_QUICKSTART.md` (multi-node)
3. Experiment with different models and batch sizes
4. Modify code to add custom models
5. Combine strategies (pipeline + ZeRO hybrid)

---

## 🔬 Tested Results

### ZeRO Performance (GPT-2 Medium, 355M params)

| Strategy | Time/Epoch | GPU Memory | Speedup | Memory Saved |
|----------|-----------|------------|---------|--------------|
| Data Parallel | 125s | 13.62 GB | 1.0× | 0% |
| ZeRO-1 | 52s | 9.64 GB | 2.4× | 29% ✅ |
| ZeRO-2 | 52s | 10.55 GB | 2.4× | 23% ✅ |
| ZeRO-3 | 105s | 21.03 GB | 1.2× | -54% |
| ZeRO-Offload | 375s | 10.10 GB | 0.3× | 26% |

**Key insight:** ZeRO-2 provides best balance of speed and memory!

### PipeDream Simulation (Educational)

| Metric | Naive Pipeline | With Microbatches | PipeDream |
|--------|---------------|-------------------|-----------|
| GPU Utilization | 25% | 75% | 90% |
| Idle Time | 75% | 25% | 10% |
| Throughput | 1.0× | 3.0× | 3.6× |

**Key insight:** Microbatches dramatically improve GPU utilization!

See `TRAINING_RESULTS.md` and `pipedream_tutorial/TEST_RESULTS.md` for complete details.

---

## 💡 When to Use What?

### Use ZeRO When:
- ✅ Model fits on 1 GPU but optimizer states don't
- ✅ You want faster training with minimal code changes
- ✅ You have 2+ GPUs available
- ✅ Model size is 100M - 10B parameters

**Start with:** ZeRO-2 (best speed/memory balance)

### Use Pipeline Parallelism When:
- ✅ Model is too large for any single GPU
- ✅ Model has clear layer-wise structure
- ✅ You have fast inter-GPU communication (NVLink)
- ✅ Model size is 10B+ parameters

**Start with:** PipeDream tutorial to learn concepts

### Use Hybrid (Pipeline + ZeRO) When:
- ✅ Training massive models (100B+ parameters)
- ✅ You have many GPUs (64+)
- ✅ Want maximum efficiency

**Examples:** GPT-3, Megatron-LM, BLOOM

---

## 🛠️ Installation

### System Requirements
- **GPU:** NVIDIA GPU with CUDA support (or CPU for learning)
- **Python:** 3.8+
- **CUDA:** 11.8+ (for GPU)
- **Memory:** 16GB+ RAM

### Install Dependencies

```bash
# Clone repository
git clone https://github.com/justin-aj/ml-sys.git
cd ml-sys/distributed-training

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install DeepSpeed and other dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
```

### For Multi-GPU Setup
See `CLUSTER_QUICKSTART.md` for detailed cluster setup instructions.

---

## 📖 Key Concepts

### Data Parallelism
- Same model replicated on each GPU
- Different data batches processed in parallel
- Gradients synchronized across GPUs

### ZeRO (Zero Redundancy Optimizer)
- **Stage 1:** Shard optimizer states (4× memory reduction)
- **Stage 2:** Shard gradients too (8× reduction) ← **Recommended**
- **Stage 3:** Shard model parameters (64× reduction)
- **Offload:** Use CPU memory for optimizer
- **Infinity:** Use NVMe storage for massive models

### Pipeline Parallelism
- Split model layers across GPUs
- Data flows through pipeline stages
- **Microbatches:** Keep all GPUs busy
- **Weight Versioning:** Ensure gradient correctness

---

## 🎨 Visual Resources

The PipeDream tutorial includes 5 visual diagrams:

1. **naive_pipeline.png** - Shows 25% GPU utilization problem
2. **microbatch_forward.png** - Shows 75% utilization with microbatches
3. **microbatch_backward.png** - Gradient flow visualization
4. **weight_versioning.png** - Weight versioning concept
5. **utilization_comparison.png** - Performance comparison chart

Generate them with:
```bash
cd pipedream_tutorial
python pipedream_visual.py
```

---

## 🔧 Configuration

### ZeRO Tutorial
Edit `real_model_example.py` CONFIG dictionary (~line 690):

```python
CONFIG = {
    "model": "gpt2-medium",      # gpt2, gpt2-medium, gpt2-large, gpt2-xl
    "strategy": "zero2",         # dp, zero1, zero2, zero3, offload, infinity
    "batch_size": 4,             # Adjust based on GPU memory
    "epochs": 2,
    "max_length": 512,
    "num_samples": 1000,         # -1 for full dataset
}
```

### PipeDream Tutorial
Edit `pipedream_tutorial/pipedream_simple.py` CONFIG:

```python
CONFIG = {
    "num_stages": 4,             # Number of pipeline stages (GPUs)
    "num_microbatches": 4,       # Microbatches per batch
    "layers_per_stage": 3,       # Layers per GPU
    "verbose": True,             # Show detailed output
}
```

---

## 📝 Common Issues

### Out of Memory (OOM)
```bash
# Solution 1: Reduce batch size
CONFIG["batch_size"] = 2

# Solution 2: Use ZeRO-Offload
CONFIG["strategy"] = "offload"

# Solution 3: Enable gradient checkpointing
# (See REAL_MODELS_GUIDE.md)
```

### Slow Training
```bash
# Solution 1: Use ZeRO-2 (fastest)
CONFIG["strategy"] = "zero2"

# Solution 2: Increase batch size
CONFIG["batch_size"] = 8

# Solution 3: Use fewer samples for testing
CONFIG["num_samples"] = 100
```

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check versions
python -c "import torch; import deepspeed; import transformers"
```

---

## 🚀 Next Steps

1. **Start with ZeRO tutorial:**
   - Run Data Parallelism baseline
   - Run ZeRO-2 and compare
   - Read TRAINING_RESULTS.md

2. **Learn PipeDream concepts:**
   - Run pipedream_simple.py
   - Generate visualizations
   - Study the timelines

3. **Experiment:**
   - Try different models (GPT-2 variants)
   - Test different ZeRO stages
   - Modify CONFIG settings

4. **Scale up:**
   - Read CLUSTER_QUICKSTART.md
   - Deploy on multi-GPU cluster
   - Combine strategies for massive models

---

## 📚 Additional Resources

### Documentation Files
- `README.md` - ZeRO tutorial main guide
- `TRAINING_RESULTS.md` - Performance benchmarks
- `REAL_MODELS_GUIDE.md` - Advanced GPT-2 training
- `CLUSTER_QUICKSTART.md` - Multi-node setup
- `pipedream_tutorial/README.md` - Pipeline parallelism
- `pipedream_tutorial/QUICKSTART.md` - 5-minute start
- `pipedream_tutorial/COMPARISON.md` - Strategy comparison

### External Resources
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [PipeDream Paper](https://arxiv.org/abs/1806.03377)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

---

## 🎓 Credits

**Created by:** Justin AJ  
**Repository:** [github.com/justin-aj/ml-sys](https://github.com/justin-aj/ml-sys)  
**Last Updated:** November 15, 2025

**Tested on:** College GPU cluster with NVIDIA GPUs

---

## ✅ Summary

This repository provides:
- ✅ 6 ZeRO strategies with real GPT-2 models
- ✅ Pipeline parallelism tutorial with visualizations
- ✅ Tested performance results
- ✅ Hands-on code examples
- ✅ Comprehensive documentation
- ✅ Works on 1 GPU or multi-GPU clusters

**Start learning distributed training today!** 🚀

---

**Quick Links:**
- [ZeRO Tutorial](README.md)
- [PipeDream Tutorial](pipedream_tutorial/README.md)
- [Strategy Comparison](pipedream_tutorial/COMPARISON.md)
- [Test Results](TRAINING_RESULTS.md)
