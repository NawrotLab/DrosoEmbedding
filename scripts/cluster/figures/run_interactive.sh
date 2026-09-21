#!/bin/bash
#SBATCH --job-name=fig23_checks
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1        # TEMPORARY: land on srv-5's H200 right away
#SBATCH --nodelist=agmn-srv-5    # TEMPORARY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Fig 2 / Fig 3 drift check: fix the stale run-10 evaluation, compare the
# current evaluation/ results against the old _chkpt_finals ones, and re-plot.
#
# Submit from the repo root (the SBATCH log paths above are relative):
#     cd ~/DrosoEmbedding && sbatch scripts/cluster/figures/run_interactive.sh
# Only step 1 needs the GPU; the rest is CPU work.

set -e

# slurmd copies this script to its own spool dir and runs it there --
# neither BASH_SOURCE nor the job's CWD/$SLURM_SUBMIT_DIR reliably point
# at the repo on this cluster. $HOME does, so anchor on that instead.
BASE_DIR="$HOME/DrosoEmbedding"
cd "$BASE_DIR"
source "$BASE_DIR/.venv/bin/activate"

# Paths come from the repo config (so DROSO_* overrides and .env are honoured),
# not hardcoded.
cfg() { python3 -c "from src.utils.config_loader import load_config; print(load_config()['paths']['$1'])" | tail -n1; }
OLD=$(cfg checkpoints_dir)     # old, hand-built _chkpt_finals/
EVAL=$(cfg eval_base_dir)      # current evaluation/ (all tasks)
EV16="${EVAL%/}/State_Modality_Valence_16"
echo "old: $OLD"
echo "new: $EVAL"

# ── 1. Re-evaluate C16_E16_H16_10 ────────────────────────────────────────────
# Its pkl scored 0.059 accuracy (old result: 0.772), well below chance, and it
# alone shifts Fig 3 panels a (16-class) and d. run_evaluation returns an
# existing pkl untouched, so the stale one has to be moved aside first.
echo "=== 1. re-evaluate C16_E16_H16_10"
STALE="$EV16/C16_E16_H16_10_evalResults.pkl"
if [ -f "$STALE" ]; then
    mkdir -p "$HOME/stale_pkls"
    mv "$STALE" "$HOME/stale_pkls/"
    echo "moved stale pkl to $HOME/stale_pkls/"
fi
( export TASK=State_Modality_Valence_16 RUN_ID=C16_E16_H16_10 CNN_DIM=16 TRF_DIM=16
  python3 -m scripts.run_evaluation )
python3 - "$STALE" <<'EOF'
import pickle, sys
acc = pickle.load(open(sys.argv[1], 'rb'))['accuracy']
print(f'C16_E16_H16_10 accuracy after re-evaluation: {acc:.4f} (old result: 0.7716)')
if acc < 0.3:
    sys.exit('Still far below the old result -> the checkpoint itself is the problem, '
             'not a stale pkl. Stopping before re-plotting.')
EOF

# ── 2. Old vs new, per run, all transformer sizes ────────────────────────────
# H16 is Fig 3 / the best-model panels; H4/H8/H32/H64 feed Fig 2's
# accuracy-vs-dimension panel.
echo "=== 2. old vs new comparison"
for T in 16 4 8 32 64; do
    echo "--- H$T"
    python3 -m scripts.analysis.compare_old_vs_new_eval --old-dir "$OLD" --new-dir "$EVAL" --trf $T --top 5
done

# ── 3. Rebuild the aggregate cache and re-plot ───────────────────────────────
# Cache has no staleness check, so it must be rebuilt after run 10 changed.
echo "=== 3. re-plot Fig 3 (rebuilds the cache) and Fig 2"
EVAL_CACHE_RECOMPUTE=true python3 -m scripts.figures.run_figure_accuracy_error
python3 -m scripts.figures.run_figure_latent
echo "=== done"
