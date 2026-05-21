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

# Optional: Activate virtualenv
source /rhomes/aabdel/DrosoEmbedding/.venv/bin/activate


# Run evaluation
# python -m scripts.figures.run_figure_overview
# python -m scripts.figures.run_figure_accuracy
# python -m scripts.figures.run_figure_latent
# python -m scripts.figures.run_figure_interpretability
# python -m scripts.figures.run_figure_exp_design
# python -m scripts.figures.run_figure_latent_marginals
# python -m scripts.figures.run_sfigure_error_structure
# python -m scripts.figures.run_sfigure_training_curves
# python -m scripts.figures.run_sfigure_latent_interactions
# python -m scripts.analysis.run_analysis_state_valence_interaction
# python -m scripts.analysis.run_neuropil_importance
# python -m scripts.analysis.run_viz_cam
# python -m scripts.diagnostics.diag_gradcam_neuropil_importance
# python -m scripts.diagnostics.diag_neuropil_methods_comparison
# python -m scripts.run_plotting
python -m scripts.figures.run_sfigure_error_structure
