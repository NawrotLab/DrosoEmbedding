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
# python -m scripts.run_figure_overview
# python -m scripts.run_figure_hAccuracy
# python -m scripts.run_figure_hLatentNew
# python -m scripts.neuropils_importance_cofficients
# python -m scripts.run_figure_interpretability
# python -m scripts.run_figure_expDesign
# python -m scripts.run_figure_latent_marginals
# python -m scripts.run_analysis_state_valence_interaction
# python -m scripts.diag_gradcam_neuropil_importance 
# python -m scripts.diag_neuropil_methods_comparison
# python -m scripts.run_Sfigure_perClassMetrics
# python -m scripts.run_figure_S4_errorStructure
python -m scripts.run_Sfigure_errorStructure_v2

# python -m scripts.neuropils_importance_cofficients
# python -m scripts.VizCAM
# python -m scripts.run_plotting
# python -m tmp.latentCentroids_tries

