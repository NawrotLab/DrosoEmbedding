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
    "Seed2lr0001_trf4 MetabolicState_2 256 0.0001 1000 4"
    "Seed2lr0001_trf8 MetabolicState_2 256 0.0001 1000 8"
    "Seed2lr0001_trf16 MetabolicState_2 256 0.0001 1000 16"
    "Seed2lr0001_trf32 MetabolicState_2 256 0.0001 1000 32"
    "Seed2lr0001_trf64 MetabolicState_2 256 0.0001 1000 64"

    "SpeedSeed6_trf4 State_Modality_6 256 0.001 2000 4"
    "SpeedSeed6_trf8 State_Modality_6 256 0.001 2000 8"
    "SpeedSeed6_trf16 State_Modality_6 256 0.001 2000 16"
    "SpeedSeed6_trf32 State_Modality_6 256 0.001 2000 32"
    "SpeedSeed6_trf64 State_Modality_6 256 0.001 2000 64"

    "SpeedSeed16_trf4 State_Modality_Valence_16 256 0.001 3000 4"
    "SpeedSeed16_trf8 State_Modality_Valence_16 256 0.001 3000 8"
    "SpeedSeed16_trf16 State_Modality_Valence_16 256 0.001 3000 16"
    "SpeedSeed16_trf32 State_Modality_Valence_16 256 0.001 3000 32"
    "SpeedSeed16_trf64 State_Modality_Valence_16 256 0.001 3000 64"
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
#SBATCH --mem=16G
#SBATCH --cpus-per-task=12
#SBATCH --gres=shard:10
#SBATCH --array=1-2
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
