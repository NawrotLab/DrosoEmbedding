"""
Supplementary Figure — Training & Validation Curves
====================================================
Two-panel figure showing mean ± 1 SD across all 50 H16 runs per task.

Panel a (top):  training loss (solid) + validation loss (dashed) with shaded bands
Panel b (bottom): validation accuracy (dashed) with shaded band

Usage (from repo root):
    python scripts/figures/run_sfigure_training_curves.py

Output:
    results/CombiPlots/{pdfs,pngs}/figS_training_curves.{pdf,png}
"""

import gc
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.visualization.figure_base import apply_style, FONT_SIZES, FIGURE_WIDTH, save_figure, add_panel_label
from src.utils.config_loader import load_config

apply_style()

# ── Configuration ─────────────────────────────────────────────────────────────
_config = load_config()
BASE_RESULTS_DIR = Path(_config['paths']['eval_base_dir'])
OUT_DIR          = Path(_config['paths']['output_dir'])
FIGURE_STEM      = 'figS_training_curves'

SHORT_RUN_THRESHOLD  = 500   # epochs — flagged to console but included
MIN_RUNS_FOR_XLIM    = 10    # clip x-axis to last epoch with ≥ this many runs
X_MAX_HARD           = 1000  # absolute x-axis ceiling
Y_MAX_LOSS           = 3.0
Y_MAX_ACC            = 100

TASKS = [
    {
        'name':       'i.  State',
        'folder':     'MetabolicState_2',
        'pkl_prefix': 'C2_E16_H16_',
        'color':      '#0072B2',
    },
    {
        'name':       'ii.  State × Modality',
        'folder':     'State_Modality_6',
        'pkl_prefix': 'C6_E16_H16_',
        'color':      '#D55E00',
    },
    {
        'name':       'iii.  State × Modality × Valence',
        'folder':     'State_Modality_Valence_16',
        'pkl_prefix': 'C16_E16_H16_',
        'color':      '#009E73',
    },
]

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_runs(task):
    """Load all pkl runs for a task; return dict of metric → (n_runs × max_epochs) array."""
    runs_dir = BASE_RESULTS_DIR / task['folder']
    pkls = sorted(runs_dir.glob(f"{task['pkl_prefix']}*_evalResults.pkl"))
    if not pkls:
        print(f"[WARN] No pkl files found in {runs_dir} with prefix '{task['pkl_prefix']}'")
        return {}

    raw = {k: [] for k in ('train_loss', 'val_loss', 'train_acc', 'val_acc')}
    for pkl_path in pkls:
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            n_epochs = len(data.get('train_loss') or [])
            if n_epochs < SHORT_RUN_THRESHOLD:
                print(f"  [FLAG] Short run ({n_epochs} epochs): {pkl_path.name}")
            for k in raw:
                arr = data.get(k)
                if arr is not None:
                    raw[k].append(np.asarray(arr, dtype=float))
            del data
            gc.collect()
        except Exception as e:
            print(f"  [WARN] {pkl_path.name}: {e}")

    # Pad shorter runs with NaN up to the longest run; use nanmean/nanstd at plot time
    result = {}
    for k, arrays in raw.items():
        if not arrays:
            continue
        max_len = max(len(a) for a in arrays)
        padded = np.full((len(arrays), max_len), np.nan)
        for i, a in enumerate(arrays):
            padded[i, :len(a)] = a
        result[k] = padded  # (n_runs, max_len)
    return result


# ── Load all tasks and compute x-axis clip ────────────────────────────────────

all_mats = {}
for task in TASKS:
    print(f"\nLoading {task['folder']} …")
    mats = _load_runs(task)
    all_mats[task['folder']] = mats

    if not mats or 'train_loss' not in mats:
        continue

    mat      = mats['train_loss']
    lengths  = (~np.isnan(mat)).sum(axis=1)  # actual epoch count per run
    n_contrib_final = int((~np.isnan(mat[:, -1])).sum())
    print(f"  {task['name']} | n_runs={len(lengths)} | "
          f"mean={lengths.mean():.0f}±{lengths.std():.0f} | "
          f"min={int(lengths.min())} | max={int(lengths.max())} | "
          f"runs at final epoch={n_contrib_final}")

# Diagnostic: last epoch where ≥ MIN_RUNS_FOR_XLIM runs still contribute
for task in TASKS:
    mat = all_mats.get(task['folder'], {}).get('train_loss')
    if mat is None:
        continue
    n_contrib = (~np.isnan(mat)).sum(axis=0)
    valid_epochs = np.where(n_contrib >= MIN_RUNS_FOR_XLIM)[0]
    clip = int(valid_epochs[-1]) if len(valid_epochs) else 0
    print(f"  {task['folder']}: ≥{MIN_RUNS_FOR_XLIM} runs up to epoch {clip}")

x_max = X_MAX_HARD

# ── Figure ────────────────────────────────────────────────────────────────────

fig, (ax_loss, ax_acc) = plt.subplots(
    2, 1, figsize=(FIGURE_WIDTH, 9), sharex=True,
    gridspec_kw=dict(hspace=0.25),
)
fig.subplots_adjust(left=0.12, right=0.97, top=0.94, bottom=0.09)

legend_task_handles = []

for task in TASKS:
    col  = task['color']
    mats = all_mats.get(task['folder'], {})
    if not mats:
        continue

    for metric, mat in mats.items():
        n_epochs = mat.shape[1]
        epochs   = np.arange(n_epochs)
        mean     = np.nanmean(mat, axis=0)
        std      = np.nanstd(mat, axis=0)
        is_val   = metric.startswith('val')
        ls       = '--' if is_val else '-'

        if 'loss' in metric:
            ax = ax_loss
        else:
            if not is_val:
                continue  # panel b: val_acc only
            ax = ax_acc

        ax.plot(epochs, mean, color=col, linewidth=1.8, linestyle=ls)
        ax.fill_between(epochs, mean - std, mean + std,
                        color=col, alpha=0.2, linewidth=0)

    legend_task_handles.append(
        mlines.Line2D([], [], color=col, linewidth=2.0, label=task['name'])
    )

# ── Axis limits and spine clipping ────────────────────────────────────────────
ax_loss.set_xlim(0, x_max)
ax_loss.set_ylim(0, Y_MAX_LOSS)
ax_acc.set_xlim(0, x_max)
ax_acc.set_ylim(0, Y_MAX_ACC)

for ax, ymin, ymax in [(ax_loss, 0, Y_MAX_LOSS), (ax_acc, 0, Y_MAX_ACC)]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_bounds(ymin, ymax)
    ax.spines['bottom'].set_bounds(0, x_max)
    ax.tick_params(axis='both', labelsize=FONT_SIZES['tick'])

ax_loss.set_ylabel('Cross-Entropy Loss',  fontsize=FONT_SIZES['label'])
ax_acc.set_ylabel('Validation Accuracy',  fontsize=FONT_SIZES['label'])
ax_acc.set_xlabel('Epoch',                fontsize=FONT_SIZES['label'])

# Panel labels (outside left spine via axes-fraction transform)
add_panel_label(fig, 'a', ax=ax_loss, dx=-0.06, dy=0.02)
add_panel_label(fig, 'b', ax=ax_acc,  dx=-0.06, dy=0.02)

# ── Legend in panel b, lower-left ─────────────────────────────────────────────
style_handles = [
    mlines.Line2D([], [], color='0.4', linewidth=1.5, linestyle='-',  label='Training'),
    mlines.Line2D([], [], color='0.4', linewidth=1.5, linestyle='--', label='Validation'),
]
ax_acc.legend(
    handles=legend_task_handles + style_handles,
    fontsize=FONT_SIZES['legend'],
    frameon=False,
    loc='lower right',
)

# ── Save ──────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
save_figure(fig, str(OUT_DIR / f'{FIGURE_STEM}.pdf'))
