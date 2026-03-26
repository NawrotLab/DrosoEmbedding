#!/bin/bash
#SBATCH --job-name=viz_model
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Optional: Activate virtualenv
source /rhomes/aabdel/DrosoEmbedding/.venv/bin/activate

# Run evaluation
# python -m scripts.run_figure_hLatent
# python -m scripts.neuropils_importance_cofficients
# python -m scripts.run_figure_interpretability
# python -m scripts.VizCAM
# python -m scripts.run_plotting
python -m scripts.run_figure_hLatentNew
# python -m scripts.run_figure_overview
# python -m tmp.latentCentroids_tries

