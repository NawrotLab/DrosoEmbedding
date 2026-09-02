#!/bin/bash
#SBATCH --job-name=ko_permutation
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodelist=agmn-srv-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# slurmd copies this script to its own spool dir before running it, so
# BASH_SOURCE would point there instead of the repo -- use the directory
# sbatch was invoked from instead.
BASE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$BASE_DIR/.venv/bin/activate"

cd "$BASE_DIR"

# python -m scripts.analysis.diag_static_fill_viz                      # baseline fill diagnostic
python -m scripts.analysis.diag_KO_quality --variant shuffled
python -m scripts.analysis.diag_KO_permutation --variant shuffled
python -m scripts.analysis.diag_KO_contrast_comparison
# python -m scripts.analysis.diag_KO_quality --variant static
# python -m scripts.analysis.diag_KO_permutation --variant static
# python -m scripts.analysis.diag_KO_quality --variant noisefill
# python -m scripts.analysis.diag_KO_permutation --variant noisefill
# python -m scripts.analysis.diag_KO_quality                           # zero-fill
# python -m scripts.analysis.diag_KO_permutation                       # zero-fill
