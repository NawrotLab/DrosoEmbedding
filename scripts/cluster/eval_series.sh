#!/bin/bash

# Simple script to submit multiple evaluation jobs with different configurations

# Base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BASE_DIR"

# Create a timestamp for this batch of experiments
TIMESTAMP=$(date +%Y%m%d)
BATCH_DIR="logs/experiments/$TIMESTAMP"
mkdir -p "$BATCH_DIR"

# Define your experiment configurations
# Format: RUN_ID
CONFIGURATIONS=(
    "C2_E16_H16_43"
    "C2_E16_H16_48"
    "C2_E16_H32_21"
)


# Submit each configuration as a separate job
for config in "${CONFIGURATIONS[@]}"; do
    # Parse configuration
    read -r RUN_ID <<< "$config"
    
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
#SBATCH --cpus-per-task=4
#SBATCH --gres=shard:6
#SBATCH --partition=interactive 

# Environment variables
export RUN_ID="${RUN_ID}"

# Change to project directory
cd "$BASE_DIR"

# Activate environment
source "$BASE_DIR/.venv/bin/activate"

# Run training
python -m scripts.run_evaluation --run_name "${RUN_ID}"
EOF

    # Make the job script executable, # gpu:a6000:1; # gpu
    chmod +x "$JOB_SCRIPT"
    
    # Submit the job
    echo "Submitting job: $JOB_NAME"
    sbatch "$JOB_SCRIPT"
    
    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

echo "Submitted ${#CONFIGURATIONS[@]} jobs in batch $TIMESTAMP"
echo "Logs directory: $BATCH_DIR"
