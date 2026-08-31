#!/bin/bash

#SBATCH --job-name=train_model
#SBATCH --time=130:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15GB
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodelist=agmn-srv-5
#SBATCH --array=9
#SBATCH --output=logs/training/%j-%x.out
#SBATCH --error=logs/training/%j-%x.err


BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$BASE_DIR/.venv/bin/activate"

RUN_NAME="${SLURM_ARRAY_TASK_ID}"

python -m scripts.training --run_name $RUN_NAME
