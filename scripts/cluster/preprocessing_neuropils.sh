#!/bin/bash

# Simple script to submit multiple preprocessing jobs with different configurations
# Run this directly from your terminal (don't use sbatch)

# Base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BASE_DIR"

# Create a timestamp for this batch of experiments
TIMESTAMP=$(date +%Y%m%d)
BATCH_DIR="logs/experiments/$TIMESTAMP"
mkdir -p "$BATCH_DIR"

# Define your experiment configurations
# Format: RUN_ID METHOD_CH TIMES NEUROPIL [REMOVE_NEUROPIL] [RECORDINGS_FILE]
# REMOVE_NEUROPIL defaults to false; RECORDINGS_FILE is optional (limits which recordings are processed)
CONFIGURATIONS=(
    # Example configurations - uncomment and modify as needed
    # "pre1_AL meanZ logTs AL"
    # "pre1_AL meanZ allTs AL"
    # "pre2_MB meanZ logTs MB"
    # "pre2_MB meanZ allTs MB"
    # "pre3_PENP meanZ logTs PENP"
    # "pre3_PENP meanZ allTs PENP"
    # "pre5_CX meanZ logTs CX"
    # "pre5_CX meanZ allTs CX"
    # "pre7_LX meanZ logTs LX"
    # "pre7_LX meanZ allTs LX"
    # "pre8_SNP meanZ logTs SNP"
    # "pre8_SNP meanZ allTs SNP"
    # "pre9_INP meanZ logTs INP"
    # "pre9_INP meanZ allTs INP"
    # "pre10_LH meanZ logTs LH"
    # "pre10_LH meanZ allTs LH"
    # "pre12_VMNP meanZ logTs VMNP"
    # "pre12_VMNP meanZ allTs VMNP"

    # KO preprocessing — missing/incomplete neuropils (initial run)
    # "KO_VLNP meanZ allTs VLNP true"
    # "KO_OL   meanZ allTs OL   true"
    # "KO_GNG  meanZ allTs GNG  true ${BASE_DIR}/results/diagnostics/KO_permutation/missing_recordings_KO_GNG.txt"

    # KO preprocessing — targeted fixes from diag_KO_quality
    "KO_GNG_fix  meanZ allTs GNG  true ${BASE_DIR}/scripts/cluster/recording_filters/KO_GNG_fix.txt"
    "KO_VLNP_fix meanZ allTs VLNP true ${BASE_DIR}/scripts/cluster/recording_filters/KO_VLNP_OL_fix.txt"
    "KO_OL_fix   meanZ allTs OL   true ${BASE_DIR}/scripts/cluster/recording_filters/KO_VLNP_OL_fix.txt"
)

# Submit each configuration as a separate job
for config in "${CONFIGURATIONS[@]}"; do
    # Parse configuration
    read -r RUN_ID METHOD_CH TIMES NEUROPIL REMOVE_NEUROPIL RECORDINGS_FILE <<< "$config"
    REMOVE_NEUROPIL="${REMOVE_NEUROPIL:-false}"

    # Create job name
    JOB_NAME="${RUN_ID}"

    # Create job script
    JOB_SCRIPT="${BATCH_DIR}/${JOB_NAME}.sh"

    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${BATCH_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${BATCH_DIR}/${JOB_NAME}_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=15G
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu
#SBATCH --nodelist=agmn-srv-5

# Environment variables
export RUN_ID="${RUN_ID}"
export METHOD_CH="${METHOD_CH}"
export TIMES="${TIMES}"
export NEUROPIL="${NEUROPIL}"
export REMOVE_NEUROPIL="${REMOVE_NEUROPIL}"
$([ -n "${RECORDINGS_FILE}" ] && echo "export RECORDINGS_FILE=\"${RECORDINGS_FILE}\"")

# Change to project directory
cd "$BASE_DIR"

# Activate environment
source "$BASE_DIR/.venv/bin/activate"

# Run preprocessing
python -m src.preprocessing.clean_dataset
EOF

    # Make the job script executable
    chmod +x "$JOB_SCRIPT"

    # Submit the job
    echo "Submitting job: $JOB_NAME (REMOVE_NEUROPIL=${REMOVE_NEUROPIL}${RECORDINGS_FILE:+, filter=$RECORDINGS_FILE})"
    sbatch "$JOB_SCRIPT"

    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

echo "Submitted ${#CONFIGURATIONS[@]} jobs in batch $TIMESTAMP"
echo "Logs directory: $BATCH_DIR"

