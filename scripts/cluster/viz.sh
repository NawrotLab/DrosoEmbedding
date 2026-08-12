#!/bin/bash
#SBATCH --job-name=viz_model
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodelist=agmn-srv-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Optional: Activate virtualenv
source /rhomes/aabdel/DrosoEmbedding/.venv/bin/activate


# Run script
# python -m scripts.analysis.diag_KO_permutation

# python -m scripts.analysis.diag_gradcam_neuropil_importance
# python -m scripts.analysis.run_viz_cam
# python -m scripts.analysis.run_neuropil_importance
# python -m scripts.analysis.diag_preprocessing_KO

# python -m scripts.analysis.diag_KO_data_coverage

# python -m scripts.figures.run_sfigure_latent_interactions
# python -m scripts.figures.run_sfigure_training_curves
python -m scripts.figures.run_figure_accuracy_error
# python -m scripts.figures.run_sfigure_exp_design
# python -m scripts.figures.run_figure_overview
# python -m scripts.figures.run_figure_latent

# python -m scripts.figures.run_sfigure_ko_neuropils --norm 2d
# python -m scripts.figures.run_figure_gradcam_neuropils
