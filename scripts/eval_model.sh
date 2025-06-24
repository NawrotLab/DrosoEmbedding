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

eval "$(conda shell.bash hook)"
conda activate myenv

python -m scripts.evaluating

# After completion:
DATE=$(date +%Y-%m-%d)
LOG_DIR="logs/slurm/${DATE}"
mkdir -p "$LOG_DIR"
mv "logs/slurm/eval_model-$SLURM_JOB_ID.out" "$LOG_DIR/"
mv "logs/slurm/eval_model-$SLURM_JOB_ID.err" "$LOG_DIR/"