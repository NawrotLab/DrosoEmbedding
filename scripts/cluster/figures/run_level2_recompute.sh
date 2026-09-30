#!/bin/bash
#SBATCH --job-name=level2_recompute
#SBATCH --time=1-12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodelist=agmn-srv-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Level-2 proof: force Fig 4 and Fig S4 to recompute from scratch (checkpoints +
# preprocessed frames + GPU), rather than loading the published results/. Real
# resource_log timing gets printed for both (see [resource] lines in the log).
#
# Measured on an H200: Fig 4 ~2 min, Fig S4 ~17.7 hours (50 checkpoints x 12
# neuropils, ~20 min each). Hence the 36h limit -- a 12h limit gets S4 killed
# at ~35/50 checkpoints, and there is no resume, so a killed job starts over.
#
# Backs up the existing published results/*.npz first (by timestamp) so this
# is non-destructive and the load-path can still be compared against afterwards.
#
# Submit from the repo root:
#     cd ~/DrosoEmbedding && sbatch scripts/cluster/figures/run_level2_recompute.sh

set -x
BASE_DIR="$HOME/DrosoEmbedding"
cd "$BASE_DIR"
source "$BASE_DIR/.venv/bin/activate"

echo "branch: $(git branch --show-current)   commit: $(git log -1 --oneline)"

cfg() { python3 -c "from src.utils.config_loader import load_config; print(load_config()['paths']['$1'])" | tail -n1; }
RESULTS_DIR=$(cfg results_dir)
echo "results dir: $RESULTS_DIR"

TS=$(date +%Y%m%d_%H%M%S)
for f in gradcam_neuropils_conv1.npz ko_static.npz; do
    if [ -f "$RESULTS_DIR/$f" ]; then
        cp "$RESULTS_DIR/$f" "$RESULTS_DIR/${f%.npz}_backup_$TS.npz"
        echo "backed up $f -> ${f%.npz}_backup_$TS.npz"
    fi
done

echo "=== Fig 4 --recompute (GradCAM: forward + backward pass, one model, test set) ==="
time python3 -m scripts.figures.run_fig4_gradcam_neuropils --recompute

echo "=== S4 --recompute (50 checkpoints x 12 neuropils, forward passes only) ==="
time python3 -m scripts.figures.run_figS4_ko_neuropils --recompute

echo "=== resource summary ==="
grep -rh '\[resource\]' logs/run_fig4_gradcam_neuropils logs/run_figS4_ko_neuropils 2>/dev/null | tail -20

echo "=== done ==="
