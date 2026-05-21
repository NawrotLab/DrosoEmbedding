#!/bin/bash
#SBATCH --job-name=time
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=agmn-srv-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=logs/slurm/%j-%x.out
#SBATCH --error=logs/slurm/%j-%x.err

# Optional: Activate virtualenv
source /rhomes/aabdel/DrosoEmbedding/.venv/bin/activate

# Set run name manually or via argument

# Run evaluation
python -m tmp.dummy_timekilller
