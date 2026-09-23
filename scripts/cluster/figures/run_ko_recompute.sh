#!/bin/bash
#SBATCH --job-name=ko_recompute
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodelist=agmn-srv-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Force a full recompute of Fig S4's knockout stack (all 50 C16_E16_H16_*
# checkpoints x 12 neuropils) -- the currently-published stack has only 49
# runs (one checkpoint was missing/skipped when it was first computed,
# confirmed to match the submitted figure exactly, so NOT a reproducibility
# bug -- this is a check for whether all 50 checkpoints are now available,
# since the 3 originally-missing "best" checkpoints were recovered/published
# on 2026-09-11, likely after this stack was originally built).
#
# Submit from the repo root:
#     cd ~/DrosoEmbedding && sbatch scripts/cluster/figures/run_ko_recompute.sh
# Takes on the order of half a day (50 checkpoints x 12 knockout forward
# passes over the full test set, sequential in one process).

set -e
BASE_DIR="$HOME/DrosoEmbedding"
cd "$BASE_DIR"
source "$BASE_DIR/.venv/bin/activate"

echo "branch: $(git branch --show-current)   commit: $(git log -1 --oneline)"

cfg() { python3 -c "from src.utils.config_loader import load_config; print(load_config()['paths']['$1'])" | tail -n1; }
RESULTS_PATH="$(cfg results_dir)/ko_static.npz"
echo "results path: $RESULTS_PATH"

# Keep the current (submission-matching, 49-run) stack -- don't lose it if
# this recompute fails partway, or if the result turns out to differ from
# what was submitted and we need to compare the two.
if [ -f "$RESULTS_PATH" ]; then
    BACKUP="${RESULTS_PATH%.npz}_49run_$(date +%Y%m%d%H%M%S).npz"
    cp "$RESULTS_PATH" "$BACKUP"
    echo "backed up existing results -> $BACKUP"
fi

python3 -m scripts.figures.run_figS4_ko_neuropils --recompute

echo "=== resulting stack shape"
python3 -c "
from src.analysis.ko_permutation import load_ko_results
stack, sizes, names = load_ko_results('$RESULTS_PATH')
print(f'{stack.shape} (runs x classes x neuropils)')
print('50 runs -> all checkpoints now available' if stack.shape[0] == 50 else f'still {stack.shape[0]} runs -- see the [WARN]/Checkpoint-missing lines in the .err log for which run_id')
"
echo "=== done"
