#!/bin/bash

#SBATCH --job-name=train_model
#SBATCH --time=130:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15GB
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodelist=agmn-srv-5
#SBATCH --array=3
#SBATCH --output=logs/training/%j-%x.out
#SBATCH --error=logs/training/%j-%x.err

# Retraining the one missing/lost checkpoint for MetabolicState_2's H8
# bucket (C2_E16_H8_3) -- its .pth went missing from the cluster, and it
# was one of the three "best" models actually used for published figures.
# Edit RUN_ID/TASK/CNN_DIM/TRF_DIM (and --array above) for any other
# single ad-hoc run.
export RUN_ID="C2_E16_H8"
export TASK="MetabolicState_2"
export CNN_DIM=16
export TRF_DIM=8

# slurmd copies this script to its own spool dir and runs it there --
# neither BASH_SOURCE nor the job's CWD/$SLURM_SUBMIT_DIR reliably point
# at the repo on this cluster. $HOME does, so anchor on that instead.
BASE_DIR="$HOME/DrosoEmbedding"
source "$BASE_DIR/.venv/bin/activate"

RUN_NAME="${SLURM_ARRAY_TASK_ID}"

python -m scripts.training --run_name $RUN_NAME
