#!/bin/bash
#SBATCH --job-name=eval_model
#SBATCH --time=10:00:00
#SBATCH --partition=interactive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --gres=shard:2
#SBATCH --output=logs/slurm/%j-%x.out
#SBATCH --error=logs/slurm/%j-%x.err

# Optional: Activate virtualenv
source /rhomes/aabdel/DrosoEmbedding/.venv/bin/activate

# Set run name manually or via argument

# Run evaluation
python -m scripts.run_evaluation
python -m scripts.run_plotting
