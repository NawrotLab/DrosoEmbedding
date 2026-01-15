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
Format: RUN_ID TASK BATCH_SIZE LEARNING_RATE EPOCHS CNN_DIM TRF_DIM
CONFIGURATIONS=(

    # "C2_E16_H2 MetabolicState_2 256 0.001 1000 16 2"
    # "C2_E16_H4 MetabolicState_2 256 0.001 1000 16 4"
    # "C2_E16_H8 MetabolicState_2 256 0.001 1000 16 8"
    # "C2_E16_H16 MetabolicState_2 256 0.001 1000 16 16"
    # "C2_E16_H32 MetabolicState_2 256 0.001 1000 16 32"
    # "C2_E16_H64 MetabolicState_2 256 0.001 1000 16 64"

    # "C6_E16_H2 State_Modality_6 256 0.001 1000 16 2"
    # "C6_E16_H4 State_Modality_6 256 0.001 1000 16 4"
    # "C6_E16_H8 State_Modality_6 256 0.001 1000 16 8"
    # "C6_E16_H16 State_Modality_6 256 0.001 1000 16 16"
    # "C6_E16_H32 State_Modality_6 256 0.001 1000 16 32"
    # "C6_E16_H64 State_Modality_6 256 0.001 1000 16 64"

    # "C16_E16_H2 State_Modality_Valence_16 256 0.001 1000 16 2"
    # "C16_E16_H4 State_Modality_Valence_16 256 0.001 1000 16 4"
    # "C16_E16_H8 State_Modality_Valence_16 256 0.001 1000 16 8"
    # "C16_E16_H16 State_Modality_Valence_16 256 0.001 1000 16 16"
    # "C16_E16_H32 State_Modality_Valence_16 256 0.001 1000 16 32"
    # "C16_E16_H64 State_Modality_Valence_16 256 0.001 1000 16 64"


    #-----------


    # "C2_E2_H16 MetabolicState_2 256 0.001 1000 2 16"
    # "C2_E4_H16 MetabolicState_2 256 0.001 1000 4 16"
    "C2_E8_H16 MetabolicState_2 256 0.001 1000 8 16"
    # "C2_E32_H16 MetabolicState_2 256 0.001 1000 32 16"
    # "C2_E64_H16 MetabolicState_2 256 0.001 1000 64 16"

    # "C6_E2_H16 State_Modality_6 256 0.001 1000 2 16"
    # "C6_E4_H16 State_Modality_6 256 0.001 1000 4 16"
    # "C6_E8_H16 State_Modality_6 256 0.001 1000 8 16"
    # "C6_E32_H16 State_Modality_6 256 0.001 1000 32 16"
    # "C6_E64_H16 State_Modality_6 256 0.001 1000 64 16"
    
    # "C16_E2_H16 State_Modality_Valence_16 256 0.001 1000 2 16"
    # "C16_E4_H16 State_Modality_Valence_16 256 0.001 1000 4 16"
    # "C16_E8_H16 State_Modality_Valence_16 256 0.001 1000 8 16"
    # "C16_E32_H16 State_Modality_Valence_16 256 0.001 1000 32 16"
    # "C16_E64_H16 State_Modality_Valence_16 256 0.001 1000 64 16"
    
)


# Submit each configuration as a separate job
for config in "${CONFIGURATIONS[@]}"; do
    # Parse configuration
    read -r RUN_ID TASK BATCH_SIZE LEARNING_RATE EPOCHS CNN_DIM TRF_DIM <<< "$config"
    
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
#SBATCH --gres=gpu:a6000:1
#SBATCH --array=34-37
#SBATCH --partition=gpu

# Environment variables
export RUN_ID="${RUN_ID}"
export TASK="${TASK}"
export BATCH_SIZE=${BATCH_SIZE}
export LEARNING_RATE=${LEARNING_RATE}
export EPOCHS=${EPOCHS}
export CNN_DIM=${CNN_DIM}
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
