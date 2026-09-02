#!/bin/bash

# Submit static-fill KO preprocessing jobs for all 12 neuropils.
# Run this directly from your terminal (don't use sbatch).
# Format: "RUN_ID TIMES NEUROPIL [RECORDINGS_FILE]"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$BASE_DIR"

TIMESTAMP=$(date +%Y%m%d)
BATCH_DIR="logs/experiments/${TIMESTAMP}_staticfill"
mkdir -p "$BATCH_DIR"

CONFIGURATIONS=(
    # ── full runs (all recordings) ─────────────────────────────────────────────
    "staticfill_AL   allTs AL"
    "staticfill_MB   allTs MB"
    "staticfill_PENP allTs PENP"
    "staticfill_VLNP allTs VLNP"
    "staticfill_CX   allTs CX"
    "staticfill_GNG  allTs GNG"
    "staticfill_LX   allTs LX"
    "staticfill_SNP  allTs SNP"
    "staticfill_INP  allTs INP"
    "staticfill_LH   allTs LH"
    "staticfill_OL   allTs OL"
    "staticfill_VMNP allTs VMNP"

    # ── targeted fixes for specific recordings, if ever needed ────────────────
    # "staticfill_fix_AL allTs AL ${BASE_DIR}/scripts/cluster/preprocessing/recording_filters/staticfill_fix.txt"
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
$([ -n "${RECORDINGS_FILE}" ] && echo "export RECORDINGS_FILE=\"${RECORDINGS_FILE}\"")

cd "$BASE_DIR"
source "$BASE_DIR/.venv/bin/activate"
python -m src.preprocessing.clean_dataset_staticfill
EOF

    chmod +x "$JOB_SCRIPT"
    echo "Submitting: $RUN_ID (neuropil=$NEUROPIL, fill=t=100..250)"
    sbatch "$JOB_SCRIPT"
    sleep 1
done

echo "Submitted ${#CONFIGURATIONS[@]} staticfill jobs. Logs: $BATCH_DIR"
