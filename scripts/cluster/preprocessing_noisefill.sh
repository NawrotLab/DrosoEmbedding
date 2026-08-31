#!/bin/bash

# Submit noisefill KO preprocessing jobs for all 12 neuropils.
# Run this directly from your terminal (don't use sbatch).
# Format: "RUN_ID TIMES NEUROPIL [RECORDINGS_FILE]"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BASE_DIR"

TIMESTAMP=$(date +%Y%m%d)
BATCH_DIR="logs/experiments/${TIMESTAMP}_noisefill"
mkdir -p "$BATCH_DIR"

CONFIGURATIONS=(
    # ── full runs (all recordings) ─────────────────────────────────────────────
    "noisefill_AL   allTs AL"
    "noisefill_MB   allTs MB"
    "noisefill_PENP allTs PENP"
    "noisefill_VLNP allTs VLNP"
    "noisefill_CX   allTs CX"
    "noisefill_GNG  allTs GNG"
    "noisefill_LX   allTs LX"
    "noisefill_SNP  allTs SNP"
    "noisefill_INP  allTs INP"
    "noisefill_LH   allTs LH"
    "noisefill_OL   allTs OL"
    "noisefill_VMNP allTs VMNP"

    # ── targeted fixes (uncomment as needed) ──────────────────────────────────
    # "noisefill_fix_AL   allTs AL   ${BASE_DIR}/scripts/cluster/recording_filters/noisefill_fix.txt"
    # "noisefill_fix_MB   allTs MB   ${BASE_DIR}/scripts/cluster/recording_filters/noisefill_fix.txt"
    # "noisefill_fix_PENP allTs PENP ${BASE_DIR}/scripts/cluster/recording_filters/noisefill_fix.txt"
    # "noisefill_fix_VLNP allTs VLNP ${BASE_DIR}/scripts/cluster/recording_filters/noisefill_fix.txt"
    # "noisefill_fix_CX   allTs CX   ${BASE_DIR}/scripts/cluster/recording_filters/noisefill_fix.txt"
    # "noisefill_fix_GNG  allTs GNG  ${BASE_DIR}/scripts/cluster/recording_filters/noisefill_fix.txt"
    # "noisefill_fix_LX   allTs LX   ${BASE_DIR}/scripts/cluster/recording_filters/noisefill_fix.txt"
    # "noisefill_fix_SNP  allTs SNP  ${BASE_DIR}/scripts/cluster/recording_filters/noisefill_fix.txt"
    # "noisefill_fix_INP  allTs INP  ${BASE_DIR}/scripts/cluster/recording_filters/noisefill_fix.txt"
    # "noisefill_fix_LH   allTs LH   ${BASE_DIR}/scripts/cluster/recording_filters/noisefill_fix.txt"
    # "noisefill_fix_OL   allTs OL   ${BASE_DIR}/scripts/cluster/recording_filters/noisefill_fix.txt"
    # "noisefill_fix_VMNP allTs VMNP ${BASE_DIR}/scripts/cluster/recording_filters/noisefill_fix.txt"
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
export REMOVE_NEUROPIL="false"
$([ -n "${RECORDINGS_FILE}" ] && echo "export RECORDINGS_FILE=\"${RECORDINGS_FILE}\"")

cd "$BASE_DIR"
source "$BASE_DIR/.venv/bin/activate"
python -m src.preprocessing.clean_dataset_noisefill
EOF

    chmod +x "$JOB_SCRIPT"
    echo "Submitting: $RUN_ID (neuropil=$NEUROPIL, times=$TIMES)"
    sbatch "$JOB_SCRIPT"
    sleep 1
done

echo "Submitted ${#CONFIGURATIONS[@]} noisefill jobs. Logs: $BATCH_DIR"
