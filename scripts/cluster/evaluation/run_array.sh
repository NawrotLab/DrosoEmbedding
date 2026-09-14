#!/bin/bash
# Evaluates the full model sweep as a real SLURM array job -- one model per
# array task, read from eval_manifest.txt (build it first with
# build_manifest.sh). run_evaluation.py already skips a model if its
# evaluation/<task>/{run_id}_evalResults.pkl already exists, so this is
# safe to resubmit for just the failed/missing array indices.
#
# Usage (from repo root, on the cluster):
#   bash scripts/cluster/evaluation/build_manifest.sh
#   sbatch scripts/cluster/evaluation/run_array.sh
#
# Pinned to agmn-srv-5 -- DROSO_PUBLISH_ROOT (the G-Node clone) lives on
# /localscratch there, which is node-local, not shared across the cluster.
#
# Adjust --array below to the manifest's actual line count (wc -l
# eval_manifest.txt) if it ever changes; %20 throttles to at most 20
# concurrent array tasks -- raise/lower based on actual GPU shard capacity.

#SBATCH --job-name=eval_sweep
#SBATCH --output=logs/experiments/eval_sweep/%A_%a.out
#SBATCH --error=logs/experiments/eval_sweep/%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=15G
#SBATCH --cpus-per-task=4
#SBATCH --gres=shard:6
#SBATCH --partition=interactive
#SBATCH --nodelist=agmn-srv-5
#SBATCH --array=1-753%20

# slurmd copies this script to its own spool dir and runs it there --
# neither BASH_SOURCE nor the job's CWD/$SLURM_SUBMIT_DIR reliably point
# at the repo on this cluster. $HOME does, so anchor on that instead.
BASE_DIR="$HOME/DrosoEmbedding"
cd "$BASE_DIR"
source "$BASE_DIR/.venv/bin/activate"

MANIFEST="$BASE_DIR/scripts/cluster/evaluation/eval_manifest.txt"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
if [ -z "$LINE" ]; then
  echo "No manifest line for array index ${SLURM_ARRAY_TASK_ID} -- exiting."
  exit 1
fi

export TASK=$(echo "$LINE" | awk '{print $1}')
export RUN_ID=$(echo "$LINE" | awk '{print $2}')

echo "Array task ${SLURM_ARRAY_TASK_ID}: TASK=${TASK} RUN_ID=${RUN_ID}"
python -m scripts.run_evaluation
