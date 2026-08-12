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
source /rhomes/aabdel/DrosoEmbedding/.venv/bin/activate


python3 -m scripts.figures.run_figure_overview
python3 -m scripts.figures.run_sfigure_exp_design
python3 -m src.visualization.vizDataset_B
