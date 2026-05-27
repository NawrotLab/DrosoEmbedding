"""
Supplementary Figure — Training & Validation Curves (v2)
=========================================================
Two-panel figure showing mean ± 1 SD across all 50 H16 runs per task.

Panel A (top):  training loss (solid) + validation loss (dashed) with shaded bands
Panel B (bottom): validation accuracy (dashed) with shaded band

Usage (from repo root):
    python scripts/figures/run_sfigure_training_curves.py

Output:
    results/CombiPlots/{pdfs,pngs}/figS_training_curves_v2.{pdf,png}
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
from src.visualization.figure_base import apply_style, FONT_SIZES, save_figure

apply_style()

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_RESULTS_DIR = Path('results') / '_chkpt_finals'
OUT_DIR          = Path('results') / 'CombiPlots'
FIGURE_STEM      = 'figS_training_curves_v2'

SHORT_RUN_THRESHOLD = 500  # epochs — flagged to console but included

TASKS = [
    {
        'name':       'i.  State (2-class)',
        'folder':     'MetabolicState_2',
        'pkl_prefix': 'C2_E16_H16_',
        'color':      '#0072B2',
    },
    {
        'name':       'ii.  State × Modality (6-class)',
        'folder':     'State_Modality_6',
        'pkl_prefix': 'C6_E16_H16_',
        'color':      '#D55E00',
    },
    {
        'name':       'iii.  State × Modality × Valence (16-class)',
        'folder':     'State_Modality_Valence_16',
        'pkl_prefix': 'C16_E16_H16_',
        'color':      '#009E73',
    },
]

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_runs(task):
    """Load all pkl runs for a task; return dict of metric → (n_runs × min_epochs) array."""
    runs_dir = BASE_RESULTS_DIR / task['folder'] / 'runs'
    pkls = sorted(runs_dir.glob(f"{task['pkl_prefix']}*.pkl"))
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

    # Truncate every run to the shortest run for that metric (no extrapolation)
    result = {}
    for k, arrays in raw.items():
        if not arrays:
            continue
        min_len = min(len(a) for a in arrays)
        result[k] = np.stack([a[:min_len] for a in arrays])  # (n_runs, min_len)
    return result


# ── Figure ────────────────────────────────────────────────────────────────────

fig, (ax_loss, ax_acc) = plt.subplots(
    2, 1, figsize=(10, 8), sharex=True,
    gridspec_kw=dict(hspace=0.06),
)
fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.09)

legend_task_handles = []

for task in TASKS:
    col = task['color']
    print(f"\nLoading {task['folder']} …")
    mats = _load_runs(task)
    if not mats:
        continue

    for metric, mat in mats.items():
        n_epochs = mat.shape[1]
        epochs   = np.arange(n_epochs)
        mean     = mat.mean(axis=0)
        std      = mat.std(axis=0)
        is_val   = metric.startswith('val')
        ls       = '--' if is_val else '-'

        if 'loss' in metric:
            ax = ax_loss
        else:
            if not is_val:
                continue  # panel B: val_acc only
            ax = ax_acc

        ax.plot(epochs, mean, color=col, linewidth=1.8, linestyle=ls)
        ax.fill_between(epochs, mean - std, mean + std,
                        color=col, alpha=0.2, linewidth=0)

    legend_task_handles.append(
        mlines.Line2D([], [], color=col, linewidth=2.0, label=task['name'])
    )

# ── Axis styling ──────────────────────────────────────────────────────────────
for ax in (ax_loss, ax_acc):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=FONT_SIZES['tick'])

ax_loss.set_ylabel('Loss',                fontsize=FONT_SIZES['label'])
ax_acc.set_ylabel('Validation Accuracy',  fontsize=FONT_SIZES['label'])
ax_acc.set_xlabel('Epoch',                fontsize=FONT_SIZES['label'])

# Panel labels
ax_loss.text(-0.07, 1.02, 'A', transform=ax_loss.transAxes,
             fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='top')
ax_acc.text(-0.07, 1.02,  'B', transform=ax_acc.transAxes,
            fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='top')

# ── Legend (task colours + solid/dashed key) ──────────────────────────────────
style_handles = [
    mlines.Line2D([], [], color='0.4', linewidth=1.5, linestyle='-',  label='Training'),
    mlines.Line2D([], [], color='0.4', linewidth=1.5, linestyle='--', label='Validation'),
]
ax_loss.legend(
    handles=legend_task_handles + style_handles,
    fontsize=FONT_SIZES['legend'],
    frameon=False,
    loc='upper right',
)

# ── Save ──────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
save_figure(fig, str(OUT_DIR / f'{FIGURE_STEM}.pdf'))
