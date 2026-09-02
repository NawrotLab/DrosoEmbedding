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
# slurmd copies this script to its own spool dir before running it, so
# BASH_SOURCE would point there instead of the repo -- use the directory
# sbatch was invoked from instead.
BASE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$BASE_DIR/.venv/bin/activate"


# ── Paper figures, in paper order ───────────────────────────────────────────

# Fig 1 — Overview
python3 -m scripts.figures.run_figure_overview

# S1 — Experimental Setup and Example Recordings
python3 -m scripts.figures.run_sfigure_exp_design

# S2 — Latent Space Interactions
python3 -m scripts.figures.run_sfigure_latent_interactions

# Fig 2 — Latent Space Visualization
python3 -m scripts.figures.run_figure_latent

# S3 — Training & Validation Curves
python3 -m scripts.figures.run_sfigure_training_curves

# Fig 3 — Classification Performance and Error Structure
python3 -m scripts.figures.run_figure_accuracy_error

# Fig 4 — Neuropil Importance (GradCAM-based)
python3 -m scripts.figures.run_figure_gradcam_neuropils

# S4 — Neuropil Knockout Importance
python3 -m scripts.figures.run_sfigure_ko_neuropils --norm 2d

# Note: run_figure_interpretability.py was an earlier, incorrect Fig 4 —
# archived in scripts/figures/archive/, superseded by Fig 4 / S4 above.

# ── Other visualization utilities (not paper figures) ──────────────────────
# python3 -m src.visualization.vizDataset_B

############################################################################
# Diagnostics / analysis (not paper figures)
# python -m scripts.analysis.diag_KO_permutation
# python -m scripts.analysis.diag_gradcam_neuropil_importance
# python -m scripts.analysis.run_viz_cam
# python -m scripts.analysis.run_neuropil_importance
# python -m scripts.analysis.diag_preprocessing_KO
# python -m scripts.analysis.diag_KO_data_coverage
