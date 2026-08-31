"""
Supplement Figure: Neuropil Knockout Importance (static-fill variant)
======================================================================
1 × 2 layout:
    ┌──────────────────────────┬──────────────────────────────┐
    │ a) Group heatmap         │ b) Contrast plots            │
    │    (8 groups × 12 np.)   │    State | Modality | Valence│
    └──────────────────────────┴──────────────────────────────┘

The KO variant used is 'static': neuropil voxels are replaced with a fixed
baseline-period mean, then mean-Z projected.  The figure shows how much each
neuropil's removal degrades accuracy, normalised by neuropil size.

Caching
-------
The first run iterates over all 50 model checkpoints (slow — ~half a day on
the server).  Results are cached to CACHE_PATH as a .npy file.  On
subsequent runs the cache is loaded and only aggregation + plotting run.
Use --recompute to force a fresh pass.

Usage (from repo root):
    python scripts/figures/run_sfigure_ko_neuropils.py
    python scripts/figures/run_sfigure_ko_neuropils.py --norm 3d
    python scripts/figures/run_sfigure_ko_neuropils.py --recompute
"""

import argparse
import os
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

from src.analysis.ko_permutation import (
    NEUROPILS, GROUP_ORDER, CONTRAST_PAIRS,
    load_or_run_ko_permutation,
    aggregate_ko_by_group,
)
from src.visualization.visualize_interpretability import (
    plot_heatmap_groups_abs,
    plot_contrasts_horizontal,
)
from src.visualization.figure_base import (
    apply_style, FONT_SIZES, FIGURE_WIDTH, save_figure, add_panel_label,
)

apply_style()

# ════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════

VARIANT    = 'static'
TASK       = 'State_Modality_Valence_16'
BASE_RUNS  = os.path.join('results', 'chkpt_runs')
RUN_PREFIX = f'{TASK}_C16_E16_H16_'
N_RUNS     = 50

CACHE_PATH = f'results/diagnostics/KO_permutation_{VARIANT}/raw_delta_stack.npy'
OUT_STEM   = f'sfig_ko_neuropils_{VARIANT}'

BATCH_SIZE  = 256
NUM_WORKERS = 4

# ════════════════════════════════════════════════
# ARGS
# ════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument('--norm', default='2d', choices=['2d', '3d', 'none'],
                    help='Size normalisation: 2d (default), 3d, or none')
parser.add_argument('--recompute', action='store_true',
                    help='Ignore cache and rerun the full 50-run permutation')
args = parser.parse_args()

size_norm = None if args.norm == 'none' else args.norm
out_stem  = f'{OUT_STEM}_norm{args.norm}'

# ════════════════════════════════════════════════
# LOAD CONFIG + TEST DATA
# ════════════════════════════════════════════════

config = load_config()
logger = setup_logger(task_name=f'sfig_ko_neuropils_{VARIANT}',
                      log_dir='logs/run_sfigure_ko_neuropils')

paths        = config['paths']
OUT_DIR      = paths['output_dir']
model_params = config['model']['parameters']

with open(f"{paths['results_root']}/config.pkl", 'rb') as fh:
    train_config = pickle.load(fh)

device      = train_config['device']
allTs_base  = train_config['paths']['allTs_path']
class_names = train_config['data']['classes']
n_classes   = len(class_names)

with open(paths['pickle_path'], 'rb') as fh:
    _, _, X_test, _, _, Y_test = pickle.load(fh)

# ════════════════════════════════════════════════
# ENUMERATE RUN DIRECTORIES
# ════════════════════════════════════════════════

all_expected = [Path(BASE_RUNS) / f'{RUN_PREFIX}{x}' for x in range(1, N_RUNS + 1)]
run_dirs     = [d for d in all_expected if d.exists()]
logger.info(f'Found {len(run_dirs)}/{N_RUNS} run directories')

# ════════════════════════════════════════════════
# LOAD OR COMPUTE KO STACK  (cached)
# ════════════════════════════════════════════════

stack = load_or_run_ko_permutation(
    cache_path=CACHE_PATH,
    recompute=args.recompute,
    logger=logger,
    # kwargs forwarded to run_ko_permutation:
    run_dirs=run_dirs,
    X_test=X_test,
    Y_test=Y_test,
    model_params=model_params,
    allTs_base=allTs_base,
    neuropils=NEUROPILS,
    variant=VARIANT,
    device=device,
    n_classes=n_classes,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)
logger.info(f'KO stack shape: {stack.shape}  '
            f'(runs × classes × neuropils)')

# ════════════════════════════════════════════════
# AGGREGATE: group profiles + contrasts
# ════════════════════════════════════════════════

rng = random.Random(42)

df_group, df_contrast = aggregate_ko_by_group(
    stack=stack,
    neuropils=NEUROPILS,
    size_norm=size_norm,
    config=config,
    rng=rng,
    logger=logger,
)

# save CSVs alongside the cache
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
df_group.to_csv(CACHE_PATH.replace('raw_delta_stack.npy',
                                    f'group_profiles_norm{args.norm}.csv'))
df_contrast.to_csv(CACHE_PATH.replace('raw_delta_stack.npy',
                                       f'contrasts_norm{args.norm}.csv'))

# ════════════════════════════════════════════════
# ASSEMBLE FIGURE
# ════════════════════════════════════════════════
#
#  ┌──────────────────────────┬──────────────────────────────┐
#  │ a) Group heatmap         │ b) Contrast plots            │
#  │    8 groups × 12 neuropils    State | Modality | Valence│
#  └──────────────────────────┴──────────────────────────────┘

logger.info('Assembling figure…')

fig = plt.figure(figsize=(FIGURE_WIDTH, 6))

gs = GridSpec(
    1, 2, figure=fig,
    left=0.05, right=0.97,
    bottom=0.18, top=0.93,
    wspace=0.35,
)

# Panel labels
add_panel_label(fig, 'a', x=0.00, y=0.95)
add_panel_label(fig, 'b', x=0.52, y=0.95)

# ── Panel a: group heatmap ────────────────────────────────────────────────

norm_label = {'2d': 'Mean ΔAccuracy per 1k px [pp / 1k px]',
              '3d': 'Mean ΔAccuracy per 10k vox [pp / 10k vox]',
              'none': 'Mean ΔAccuracy [pp]'}[args.norm]

ax_heat = fig.add_subplot(gs[0, 0])
plot_heatmap_groups_abs(
    ax=ax_heat,
    group_profiles_abs=df_group,
    neuropil_names=NEUROPILS,
    annotate=False,
)

# ── Panel b: contrast plots ───────────────────────────────────────────────

gs_contrasts = GridSpecFromSubplotSpec(
    1, 3, subplot_spec=gs[0, 1], wspace=0.06,
)
axes_contrast = [fig.add_subplot(gs_contrasts[0, i]) for i in range(3)]
plot_contrasts_horizontal(
    axes=axes_contrast,
    group_contrasts_abs=df_contrast,
    neuropil_names=NEUROPILS,
    sort_by_modality=True,
)

# ════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════

os.makedirs(OUT_DIR, exist_ok=True)
save_figure(fig, os.path.join(OUT_DIR, f'{out_stem}.pdf'),
            formats=('png', 'svg', 'pdf'))
logger.info('Done.')
