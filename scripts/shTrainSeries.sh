#!/bin/bash

# Simple script to submit multiple training jobs with different configurations
# Run this directly from your terminal (don't use sbatch)

# Base directory
BASE_DIR="/rhomes/aabdel/DrosoEmbedding"
cd "$BASE_DIR"

# Create a timestamp for this batch of experiments
TIMESTAMP=$(date +%Y%m%d)
BATCH_DIR="logs/experiments/$TIMESTAMP"
mkdir -p "$BATCH_DIR"

Define your experiment configurations
Format: RUN_ID TASK BATCH_SIZE LEARNING_RATE EPOCHS TRF_DIM
CONFIGURATIONS=(
    "C2_E16_lr01 MetabolicState_2 256 0.01 200 16"
    "C2_E16_lr001 MetabolicState_2 256 0.001 200 16"

    "C6_E16_lr01 State_Modality_6 256 0.01 200 16"
    "C6_E16_lr001 State_Modality_6 256 0.001 200 16"

    "C16_E16_lr01 State_Modality_Valence_16 256 0.01 200 16"
    "C16_E16_lr001 State_Modality_Valence_16 256 0.001 200 16"
)


# Submit each configuration as a separate job
for config in "${CONFIGURATIONS[@]}"; do
    # Parse configuration
    read -r RUN_ID TASK BATCH_SIZE LEARNING_RATE EPOCHS TRF_DIM <<< "$config"
    
    # Create job name
    JOB_NAME="${RUN_ID}"
    
    # Create job script
    JOB_SCRIPT="${BATCH_DIR}/${JOB_NAME}.sh"
    
    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${BATCH_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${BATCH_DIR}/${JOB_NAME}_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=16
#SBATCH --gres=shard:10
#SBATCH --array=3-4
#SBATCH --partition=all

# Environment variables
export RUN_ID="${RUN_ID}"
export TASK="${TASK}"
export BATCH_SIZE=${BATCH_SIZE}
export LEARNING_RATE=${LEARNING_RATE}
export EPOCHS=${EPOCHS}
export TRF_DIM=${TRF_DIM}

# Change to project directory
cd "$BASE_DIR"

# Activate environment
source /rhomes/aabdel/DrosoEmbedding/.venv/bin/activate

# Run training
python -m scripts.training --run_name "${RUN_ID}"
EOF

    # Make the job script executable
    chmod +x "$JOB_SCRIPT"
    
    # Submit the job
    echo "Submitting job: $JOB_NAME"
    sbatch "$JOB_SCRIPT"
    
    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

echo "Submitted ${#CONFIGURATIONS[@]} jobs in batch $TIMESTAMP"
echo "Logs directory: $BATCH_DIR"
