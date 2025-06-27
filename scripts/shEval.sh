#!/bin/bash
#SBATCH --job-name=eval_model
#SBATCH --time=10:00:00
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=shard:8
#SBATCH --mem=8GB
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null



source /rhomes/aabdel/DrosoEmbedding/.venv/bin/activate 

RUN_NAME="${SLURM_ARRAY_TASK_ID}"

python -m scripts.evaluating --run_name $RUN_NAME

