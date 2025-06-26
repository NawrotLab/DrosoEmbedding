#!/bin/bash

#SBATCH --job-name=train_model
#SBATCH --time=130:00:00
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --gres=shard:8
#SBATCH --array=1-3
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null


source /rhomes/aabdel/DrosoEmbedding/.venv/bin/activate 

RUN_NAME="${SLURM_ARRAY_TASK_ID}"

python -m scripts.training --run_name $RUN_NAME
