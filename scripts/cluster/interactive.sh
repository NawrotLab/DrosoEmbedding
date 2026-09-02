#!/bin/bash
#SBATCH --job-name=viz_model
#SBATCH --time=10:00:00
#SBATCH --partition=interactive
#SBATCH --gres=shard:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Optional: Activate virtualenv
# slurmd copies this script to its own spool dir before running it, so
# BASH_SOURCE would point there instead of the repo -- use the directory
# sbatch was invoked from instead.
BASE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$BASE_DIR/.venv/bin/activate"


python3 -m scripts.figures.run_figure_overview
python3 -m scripts.figures.run_sfigure_exp_design
python3 -m src.visualization.vizDataset_B
