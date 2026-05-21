#!/bin/bash
#SBATCH --job-name=eval_model
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB 
#SBATCH --gres=gpu:a6000:1 # for srv 5 specific: --nodelist=agmn-srv-5, for interactive: --gres=gpu:a6000:1
#SBATCH --output=logs/slurm/%j-%x.out
#SBATCH --error=logs/slurm/%j-%x.err

source /rhomes/aabdel/DrosoEmbedding/.venv/bin/activate

# Debug: confirm which python + which cv2 you’re using
# which python
# python -c "import sys; print(sys.executable)"
# python -c "import cv2; print('cv2 OK:', cv2.__version__)"

# Run evaluation/plotting using the venv python explicitly 
#  /rhomes/aabdel/DrosoEmbedding/.venv/bin/python -m scripts.neuropils_importance_single_neuropil_models

/rhomes/aabdel/DrosoEmbedding/.venv/bin/python -m scripts.run_evaluation
#  /rhomes/aabdel/DrosoEmbedding/.venv/bin/python -m scripts.run_plotting