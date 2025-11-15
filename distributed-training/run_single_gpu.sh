#!/bin/bash
#SBATCH --job-name=single_gpu_training
#SBATCH --partition=gpu              # GPU partition
#SBATCH --nodes=1                    # Single node
#SBATCH --ntasks=1                   # Single task
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=8            # CPUs
#SBATCH --time=04:00:00              # Longer time for single GPU
#SBATCH --mem=64G                    # Memory
#SBATCH --output=logs/single_gpu_%j.out
#SBATCH --error=logs/single_gpu_%j.err

# Create logs directory
mkdir -p logs

echo "=== SLURM Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs allocated: $SLURM_GPUS"

# Load required modules (adjust for your cluster)
module load cuda/12.1 || module load cuda
module list

# Check GPU
echo ""
echo "=== GPU Info ==="
nvidia-smi

# Activate Python environment
source ~/pt_env/bin/activate

# Run with single GPU (no distributed launcher needed)
echo ""
echo "=== Starting Training (Single GPU) ==="
python real_model_example.py

echo ""
echo "=== Training Complete ==="
