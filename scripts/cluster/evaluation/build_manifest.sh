#!/bin/bash
# Generates eval_manifest.txt: one "TASK RUN_ID" pair per line, covering the
# full sweep (5 transformer dims x 50 seeds x 3 tasks = 750) plus each
# task's one control run (3) = 753 lines total. Consumed by run_array.sh,
# which maps SLURM_ARRAY_TASK_ID -> a line number -> one model to evaluate.
#
# Usage (from repo root, on the cluster):
#   bash scripts/cluster/evaluation/build_manifest.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

MANIFEST="scripts/cluster/evaluation/eval_manifest.txt"
> "$MANIFEST"

declare -A TASK_PREFIX=(
  ["MetabolicState_2"]="C2"
  ["State_Modality_6"]="C6"
  ["State_Modality_Valence_16"]="C16"
)

for task in "${!TASK_PREFIX[@]}"; do
  prefix="${TASK_PREFIX[$task]}"
  for dim in 4 8 16 32 64; do
    for seed in $(seq 1 50); do
      echo "${task} ${prefix}_E16_H${dim}_${seed}" >> "$MANIFEST"
    done
  done
  echo "${task} ${prefix}_Ctr_E16_0" >> "$MANIFEST"
done

sort -o "$MANIFEST" "$MANIFEST"
echo "Wrote $(wc -l < "$MANIFEST") lines to $MANIFEST"
