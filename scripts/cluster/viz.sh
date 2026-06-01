#!/bin/bash
#SBATCH --job-name=viz_model
#SBATCH --time=10:00:00
#SBATCH --partition=interactive
#SBATCH --gres=shard:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --mail-user=abdelbaki.amina@icloud.com

# Optional: Activate virtualenv
source /rhomes/aabdel/DrosoEmbedding/.venv/bin/activate


# Run script
# python -m scripts.diagnostics.diag_preprocessing_KO
# python -m scripts.diagnostics.diag_KO_permutation
# python -m scripts.diagnostics.diag_KO_data_coverage

# python -m scripts.figures.run_sfigure_latent_interactions
# python -m scripts.figures.run_sfigure_training_curves
python -m scripts.figures.run_figure_accuracy_error
# python -m scripts.figures.run_sfigure_error_structure
# python -m scripts.figures.run_sfigure_expDesign
# python -m scripts.figures.run_figure_overview
# python -m scripts.figures.run_figure_latent
# python -m scripts.figures.run_figure_accuracy
# python -m scripts.figures.run_figure_interpretability