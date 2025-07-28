#!/bin/bash
#SBATCH --job-name=viz_model
#SBATCH --time=10:00:00
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8GB
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Optional: Activate virtualenv
source /rhomes/aabdel/DrosoEmbedding/.venv/bin/activate

# Run evaluation
python -m scripts.run_plotting
