#!/bin/bash
#SBATCH --job-name=results_check
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1        # TEMPORARY: land on srv-5 (where /localscratch data lives); this job itself needs no GPU
#SBATCH --nodelist=agmn-srv-5    # TEMPORARY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Level-1 "results" check: re-plot the figures that now read from the results
# (Fig 2, 3, S2, S3, S4) and compare each PNG with the one made before these
# changes. Nothing here is destructive: the results file is rebuilt in place
# (its old layout is detected as outdated), and copies of the earlier PNGs are
# kept in ~/before (never overwritten once saved).
#
# Submit from the repo root (the SBATCH log paths above are relative):
#     cd ~/DrosoEmbedding && sbatch scripts/cluster/figures/run_interactive.sh
# CPU work only, a few minutes.

# no `set -e`: one failing figure must not stop the others being checked
BASE_DIR="$HOME/DrosoEmbedding"
cd "$BASE_DIR"
source "$BASE_DIR/.venv/bin/activate"

echo "branch: $(git branch --show-current)   commit: $(git log -1 --oneline)"

cfg() { python3 -c "from src.utils.config_loader import load_config; print(load_config()['paths']['$1'])" | tail -n1; }
RESULTS_DIR=$(cfg results_dir)      # published small results (S4 stack, neuropil sizes)
PNGS=results/CombiPlots/pngs
BEFORE="$HOME/before"
FIGS="fig_accuracy_error figS_training_curves figS_ko_neuropils_static fig_latent figS_latent_interactions"
echo "results dir: $RESULTS_DIR"

# ── 0. keep the PNGs from before these changes ────────────────────────────────
mkdir -p "$BEFORE"
for f in $FIGS; do
    if [ -f "$PNGS/$f.png" ] && [ ! -f "$BEFORE/$f.png" ]; then cp "$PNGS/$f.png" "$BEFORE/"; fi
done
echo "PNGs kept in $BEFORE:"; ls -l "$BEFORE"

# ── 1. Fig S4 results: move the two small files into the results folder (once) ─
echo "=== 1. S4 results into $RESULTS_DIR"
mkdir -p "$RESULTS_DIR/ko_static"
[ -f "$RESULTS_DIR/ko_static/raw_delta_stack.npy" ] || \
    cp results/diagnostics/KO_permutation_static/raw_delta_stack.npy "$RESULTS_DIR/ko_static/"
[ -f "$RESULTS_DIR/neuropil_sizes_2d.json" ] || \
    cp results/preprocessing/neuropil_sizes_2d.json "$RESULTS_DIR/"
ls -l "$RESULTS_DIR" "$RESULTS_DIR/ko_static"

# ── 2. Re-plot from the results ──────────────────────────────────────────────
run() { echo "--- $*"; "$@" || echo "FAILED: $*"; }

echo "=== 2. re-plot (Fig 3 first: it rebuilds the outdated aggregated results, ~3 min)"
run python3 -m scripts.figures.run_figure_accuracy_error
run python3 -m scripts.analysis.describe_results
run python3 -m scripts.figures.run_sfigure_training_curves
run python3 -m scripts.figures.run_sfigure_ko_neuropils
run python3 -m scripts.figures.run_figure_latent
run python3 -m scripts.figures.run_sfigure_latent_interactions

# ── 3. Compare with the PNGs from before ─────────────────────────────────────
echo "=== 3. comparison with the PNGs from before"
for f in $FIGS; do
    if [ ! -f "$BEFORE/$f.png" ]; then echo "  $f: no earlier PNG to compare with"
    elif cmp -s "$BEFORE/$f.png" "$PNGS/$f.png"; then echo "  $f: IDENTICAL to before"
    else echo "  $f: DIFFERENT from before"; fi
done
echo "=== done"
