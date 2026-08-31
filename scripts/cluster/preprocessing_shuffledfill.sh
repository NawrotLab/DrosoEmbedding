#!/bin/bash

# Submit shuffled-baseline-fill KO preprocessing jobs for all 12 neuropils.
# Run this directly from your terminal (don't use sbatch).

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BASE_DIR"

TIMESTAMP=$(date +%Y%m%d)
BATCH_DIR="logs/experiments/${TIMESTAMP}_shuffledfill"
mkdir -p "$BATCH_DIR"

CONFIGURATIONS=(
    # ── full runs (all recordings) ─────────────────────────────────────────────
    "shuffledfill_AL   allTs AL"
    "shuffledfill_MB   allTs MB"
    "shuffledfill_PENP allTs PENP"
    "shuffledfill_VLNP allTs VLNP"
    "shuffledfill_CX   allTs CX"
    "shuffledfill_GNG  allTs GNG"
    "shuffledfill_LX   allTs LX"
    "shuffledfill_SNP  allTs SNP"
    "shuffledfill_INP  allTs INP"
    "shuffledfill_LH   allTs LH"
    "shuffledfill_OL   allTs OL"
    "shuffledfill_VMNP allTs VMNP"

    # ── targeted fixes (uncomment as needed) ──────────────────────────────────
    # "shuffledfill_fix_AL allTs AL ${BASE_DIR}/scripts/cluster/recording_filters/shuffledfill_fix.txt"
)

for config in "${CONFIGURATIONS[@]}"; do
    read -r RUN_ID TIMES NEUROPIL RECORDINGS_FILE <<< "$config"

    JOB_SCRIPT="${BATCH_DIR}/${RUN_ID}.sh"
    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${RUN_ID}
#SBATCH --output=${BATCH_DIR}/${RUN_ID}_%j.out
#SBATCH --error=${BATCH_DIR}/${RUN_ID}_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=15G
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu
#SBATCH --nodelist=agmn-srv-5

export NEUROPIL="${NEUROPIL}"
export TIMES="${TIMES}"
export METHOD_CH="meanZ"
export T_FILL_START="100"
export T_FILL_END="250"
export RANDOM_SEED="42"
$([ -n "${RECORDINGS_FILE}" ] && echo "export RECORDINGS_FILE=\"${RECORDINGS_FILE}\"")

cd "$BASE_DIR"
source "$BASE_DIR/.venv/bin/activate"
python -m src.preprocessing.clean_dataset_shuffledfill
EOF

    chmod +x "$JOB_SCRIPT"
    echo "Submitting: $RUN_ID (neuropil=$NEUROPIL, seed=42)"
    sbatch "$JOB_SCRIPT"
    sleep 1
done

echo "Submitted ${#CONFIGURATIONS[@]} shuffledfill jobs. Logs: $BATCH_DIR"
