#!/bin/bash
#SBATCH --job-name=viz_model
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodelist=agmn-srv-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Optional: Activate virtualenv
source /rhomes/aabdel/DrosoEmbedding/.venv/bin/activate


# Run script
# python3 -m scripts.figures.run_figure_overview
# python3 -m scripts.figures.run_sfigure_exp_design
# python3 -m src.visualization.vizDataset_B

# # 2, 
# python3 -m scripts.figures.run_figure_latent
# python3 -m scripts.figures.run_sfigure_latent_interactions

# 3
python3 -m scripts.figures.run_figure_accuracy_error

############################################################################
# python -m scripts.diagnostics.diag_KO_permutation

# python -m scripts.diagnostics.diag_gradcam_neuropil_importance
# python -m scripts.analysis.run_viz_camueue
# python -m scripts.analysis.run_neuropils_importance
# python -m scripts.diagnostics.diag_preprocessing_KO

# python -m scripts.diagnostics.diag_KO_data_coverage

# python -m scripts.figures.run_figure_accuracy_error

# python -m scripts.figures.run_sfigure_ko_neuropils --norm 2d
