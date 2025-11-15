#!/bin/bash

# Compare Different Strategies on Real GPT-2 Models
# This script runs the same model with different strategies and compares results

set -e

MODEL=${1:-gpt2-medium}  # Default to gpt2-medium
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)

echo "=========================================="
echo "🔬 Strategy Comparison"
echo "=========================================="
echo "Model: $MODEL"
echo "GPUs: $NUM_GPUS"
echo ""

# Create results file
RESULTS_FILE="comparison_results_${MODEL}.txt"
echo "Strategy Comparison for $MODEL on $NUM_GPUS GPUs" > $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "========================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# ============================================================================
# 1. Data Parallelism
# ============================================================================
echo ""
echo "=========================================="
echo "1️⃣  Testing Data Parallelism"
echo "=========================================="
echo ""

torchrun --nproc_per_node=$NUM_GPUS real_model_example.py \
    --model $MODEL \
    --strategy dp \
    --batch_size 4 \
    --epochs 1 \
    --num_samples 500 | tee -a $RESULTS_FILE

echo "" | tee -a $RESULTS_FILE
echo "---" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

# ============================================================================
# 2. ZeRO Stage 2
# ============================================================================
echo ""
echo "=========================================="
echo "2️⃣  Testing ZeRO Stage 2"
echo "=========================================="
echo ""

deepspeed --num_gpus=$NUM_GPUS real_model_example.py \
    --model $MODEL \
    --strategy zero2 \
    --batch_size 4 \
    --epochs 1 \
    --num_samples 500 | tee -a $RESULTS_FILE

echo "" | tee -a $RESULTS_FILE
echo "---" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

# ============================================================================
# 3. ZeRO Stage 3
# ============================================================================
echo ""
echo "=========================================="
echo "3️⃣  Testing ZeRO Stage 3"
echo "=========================================="
echo ""

deepspeed --num_gpus=$NUM_GPUS real_model_example.py \
    --model $MODEL \
    --strategy zero3 \
    --batch_size 4 \
    --epochs 1 \
    --num_samples 500 | tee -a $RESULTS_FILE

echo "" | tee -a $RESULTS_FILE
echo "---" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=========================================="
echo "📊 Comparison Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "To compare different model sizes:"
echo "  bash compare_strategies.sh gpt2        # 124M params"
echo "  bash compare_strategies.sh gpt2-medium # 355M params"
echo "  bash compare_strategies.sh gpt2-large  # 774M params"
echo "  bash compare_strategies.sh gpt2-xl     # 1.5B params"
echo ""
