# ML Systems Projects

**A comprehensive collection of GPU optimization tutorials - from graph-level rewrites to kernel fusion to auto-tuning.**

Learn how to make your ML models **2-5x faster** by optimizing at every level of the stack!

---

## 🎯 What You'll Learn

This repository contains **7 complete tutorials** teaching modern GPU optimization and distributed training:

| Tutorial | Level | Time | Speedup | What You Learn | Status |
|----------|-------|------|---------|----------------|--------|
| **[Mirage](./mirage-tutorial/)** | Graph | 1 hour | 1.5-3x | Superoptimization via equality saturation | ✅ Hands-On |
| **[TASO](./taso-tutorial/)** | Graph | 30 min | 1.5-2x | Algebraic rewrites eliminate operations | ✅ Hands-On |
| **[Mega-Kernels](./mega-kernels/)** | Kernel | 1 hour | 1.6-1.9x | CUDA kernel fusion concepts | ✅ Hands-On |
| **[Triton](./triton-tutorial/)** | Kernel | 2-3 hours | 1.3-1.5x | Production GPU programming in Python | ✅ Hands-On |
| **[Ansor](./ansor-tutorial/)** | Schedule | 30 min | 1.2-1.5x | ML-guided auto-tuning | 📖 Concept Only |
| **[Distributed Training](./distributed-training/)** | System | 2 hours | 64x memory | ZeRO, Data Parallel, Multi-GPU training | ✅ Hands-On |
| **[LLM Serving](./llm-serving/)** | Inference | 3 hours | 2-24x throughput | PagedAttention, SGLang, Speculative Decoding | ✅ Hands-On |

**Combined Impact:** Stack these techniques for **2-5x end-to-end speedup** on real models!

---

## 📋 Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (tested on Tesla V100)
- PyTorch >= 2.0.0

---

## 📄 License

MIT License - See individual project folders for details.

---

## 🤝 Contributing

Feel free to explore, learn, and adapt these examples for your own projects!
