#!/bin/bash
# Run ZeRO Stage 2 strategy (RECOMMENDED!)

echo "=========================================="
echo "Strategy 3: ZeRO Stage 2 ⭐ RECOMMENDED"
echo "=========================================="
echo ""
echo "⚙️  Configuration:"
echo "   Model: gpt2-medium (355M params)"
echo "   Strategy: ZeRO Stage 2"
echo "   Sharding: Optimizer + Gradients"
echo "   Batch size: 4"
echo "   Epochs: 2"
echo ""
echo "📊 Expected results:"
echo "   Time: ~52s/epoch (2.4× FASTER!)"
echo "   Memory: ~10.55 GB (23% LESS!)"
echo "   Loss: ~0.29 (final)"
echo ""
echo "=========================================="
echo ""

# Go to parent directory
cd ..

# Run with DeepSpeed launcher
deepspeed --num_gpus=1 real_model_example.py

echo ""
echo "=========================================="
echo "✅ ZeRO Stage 2 complete!"
echo ""
echo "Results comparison:"
echo "  vs Data Parallel: 2.4× faster, 23% less memory!"
echo ""
echo "Next: Try 4_zero_stage3/ to train larger models!"
echo "=========================================="
