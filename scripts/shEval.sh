#!/bin/bash
#SBATCH --job-name=eval_model
#SBATCH --time=10:00:00
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=shard:8
#SBATCH --mem=8GB
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Optional: Activate virtualenv
source /rhomes/aabdel/DrosoEmbedding/.venv/bin/activate

# Set run name manually or via argument
RUN_NAME="my_run_id"  # <-- CHANGE THIS!

# Run evaluation
python -m scripts.evaluating --run_name "$RUN_NAME"
