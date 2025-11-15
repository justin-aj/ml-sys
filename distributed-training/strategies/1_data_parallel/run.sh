#!/bin/bash
# Run Data Parallelism strategy

echo "=========================================="
echo "Strategy 1: Data Parallelism (Baseline)"
echo "=========================================="
echo ""
echo "⚙️  Configuration:"
echo "   Model: gpt2-medium (355M params)"
echo "   Strategy: Data Parallel (DDP)"
echo "   Batch size: 4"
echo "   Epochs: 2"
echo ""
echo "📊 Expected results:"
echo "   Time: ~125s/epoch"
echo "   Memory: ~13.62 GB"
echo "   Loss: ~0.31 (final)"
echo ""
echo "=========================================="
echo ""

# Go to parent directory
cd ..

# Run with Python (no DeepSpeed needed for DP)
python real_model_example.py

echo ""
echo "=========================================="
echo "✅ Data Parallelism complete!"
echo "Next: Try 3_zero_stage2/ for 2.4× speedup!"
echo "=========================================="
