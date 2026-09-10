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

# slurmd copies this script to its own spool dir and runs it there --
# neither BASH_SOURCE nor the job's CWD/$SLURM_SUBMIT_DIR reliably point
# at the repo on this cluster. $HOME does, so anchor on that instead.
BASE_DIR="$HOME/DrosoEmbedding"
source "$BASE_DIR/.venv/bin/activate"

# Debug: confirm which python + which cv2 you’re using
# which python
# python -c "import sys; print(sys.executable)"
# python -c "import cv2; print('cv2 OK:', cv2.__version__)"

"$BASE_DIR/.venv/bin/python" -m scripts.run_evaluation