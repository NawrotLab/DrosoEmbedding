#!/bin/bash

#SBATCH --job-name=train_model
#SBATCH --time=130:00:00
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --gres=shard:8
#SBATCH --array=1-2


source /rhomes/aabdel/DrosoEmbedding/.venv/bin/activate
# Initialize Conda and run; --nodelist=agmn-srv-4
# eval "$(conda shell.bash hook)" --> delete? 
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate myenv


# Define a unique run name based on SLURM_ARRAY_TASK_ID
RUN_NAME="run_${SLURM_ARRAY_TASK_ID}"

# Pass the run name as an argument
python -m scripts.training --run_name $RUN_NAME

# After completion:
# DATE=$(date +%Y-%m-%d)
# LOG_DIR="logs/slurm/${DATE}"
# mkdir -p "$LOG_DIR"
# mv "logs/slurm/train_model-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "$LOG_DIR/"
# mv "logs/slurm/train_model-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "$LOG_DIR/"

# Cleanup old logs
# find logs/slurm/ -type f -mtime +10 -delete

###..... SBATCH --output=logs/slurm/%x-%A_%a.out
###......SBATCH --error=logs/slurm/%x-%A_%a.err