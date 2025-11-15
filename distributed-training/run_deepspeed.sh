#!/bin/bash
#SBATCH --job-name=deepspeed_training
#SBATCH --partition=gpu              # GPU partition (adjust for your cluster)
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=4          # Number of tasks (= number of GPUs)
#SBATCH --gres=gpu:4                 # Request 4 GPUs
#SBATCH --cpus-per-task=8            # CPUs per GPU (increased for better performance)
#SBATCH --time=02:00:00              # Max runtime
#SBATCH --mem=128G                   # Total memory
#SBATCH --output=logs/deepspeed_%j.out
#SBATCH --error=logs/deepspeed_%j.err

# Print SLURM allocation info to verify you got the GPUs
echo "=== SLURM Allocation ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules (adjust versions for your cluster)
module load cuda/12.1      # Or whatever CUDA version is available
module load openmpi/4.1.5  # Or whatever OpenMPI version is available

# Print loaded modules for debugging
echo "=== Loaded Modules ==="
module list

# Print GPU info
echo ""
echo "=== Available GPUs ==="
nvidia-smi

# Print environment info
echo ""
echo "=== Environment ==="
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"

# Activate your Python environment
source ~/pt_env/bin/activate  # Adjust path to your venv

# Run with DeepSpeed launcher
echo ""
echo "=== Starting DeepSpeed Training ==="
deepspeed --num_gpus=4 real_model_example.py

echo ""
echo "=== Training Complete ==="
