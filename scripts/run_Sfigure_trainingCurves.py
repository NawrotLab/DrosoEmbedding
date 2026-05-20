"""
Supplementary Figure — Training & Validation Loss Curves
=========================================================
Three panels (one per task): training and validation loss curves across
all H16 runs (thin, translucent) with the best model highlighted (bold).

Panel a  MetabolicState_2           (2-class)
Panel b  State_Modality_6           (6-class)
Panel c  State_Modality_Valence_16  (16-class)

Usage (from repo root):
    python scripts/run_Sfigure_trainingCurves.py

Output:
    results/CombiPlots/figS_training_curves.{pdf,png}
"""

import gc
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization.figure_base import apply_style, FONT_SIZES
from src.utils.helpers import load_all_results

apply_style()

# ════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════

BASE_RESULTS_DIR = os.path.join('results', '_chkpt_finals')
OUT_DIR          = os.path.join('results', 'CombiPlots')

TASK_ORDER = [
    'MetabolicState_2',
    'State_Modality_6',
    'State_Modality_Valence_16',
]
TASK_META = {
    'MetabolicState_2':          {'label': 'i.  State (2-class)',                       'color': '#0072B2'},
    'State_Modality_6':          {'label': 'ii.  State × Modality (6-class)',            'color': '#D55E00'},
    'State_Modality_Valence_16': {'label': 'iii.  State × Modality × Valence (16-class)', 'color': '#009E73'},
}

# ════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════

print(f'Loading results index from {BASE_RESULTS_DIR} …')
results_dict = load_all_results(BASE_RESULTS_DIR, task_names=TASK_ORDER)

def _load_losses(path):
    """Load a single pkl and return (train_loss, val_loss) lists, or (None, None)."""
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        tl = data.get('train_loss')
        vl = data.get('val_loss')
        del data
        gc.collect()
        if tl and vl:
            return list(tl), list(vl)
    except Exception as e:
        print(f'  [WARN] {path}: {e}')
    return None, None


task_data = {}

for task_key in TASK_ORDER:
    if task_key not in results_dict:
        print(f'[WARN] {task_key} not found — skipping.')
        task_data[task_key] = {}
        continue

    entry = results_dict[task_key]
    meta  = TASK_META[task_key]
    print(f'\n{meta["label"]}')

    # ── Best model losses ────────────────────────────────────────────────
    best_tl = best_vl = None
    if entry.get('best_filename'):
        best_path = Path(BASE_RESULTS_DIR) / task_key / 'best' / entry['best_filename']
        best_tl, best_vl = _load_losses(best_path)
        if best_tl:
            print(f'  Best model: {len(best_tl)} epochs loaded from {entry["best_filename"]}')

    # ── H16 run losses ───────────────────────────────────────────────────
    h16_runs   = entry.get('runs', {}).get('H16', [])
    run_trains = []
    run_vals   = []
    print(f'  Loading losses from {len(h16_runs)} H16 runs …')
    for run in h16_runs:
        path = run.get('path')
        if path is None:
            continue
        tl, vl = _load_losses(path)
        if tl:
            run_trains.append(tl)
            run_vals.append(vl)
    print(f'  Loaded {len(run_trains)} run loss curves.')

    task_data[task_key] = {
        'best_train': best_tl,
        'best_val':   best_vl,
        'run_trains': run_trains,
        'run_vals':   run_vals,
    }

# ════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════

fig = plt.figure(figsize=(14, 4.5))
gs  = GridSpec(1, 3, figure=fig,
               left=0.07, right=0.98, top=0.88, bottom=0.18,
               wspace=0.32)

PANEL_LABELS = ['a', 'b', 'c']

for pi, task_key in enumerate(TASK_ORDER):
    ax   = fig.add_subplot(gs[pi])
    meta = TASK_META[task_key]
    col  = meta['color']
    d    = task_data.get(task_key, {})

    run_trains = d.get('run_trains', [])
    run_vals   = d.get('run_vals',   [])
    best_tl    = d.get('best_train')
    best_vl    = d.get('best_val')

    # Pad all run curves to the same length for alignment
    def _pad(curves):
        if not curves:
            return []
        max_len = max(len(c) for c in curves)
        return [c + [c[-1]] * (max_len - len(c)) for c in curves]

    run_trains = _pad(run_trains)
    run_vals   = _pad(run_vals)

    # ── Per-run curves (thin, translucent) ───────────────────────────────
    for tl in run_trains:
        ax.plot(tl, color=col, linewidth=0.6, alpha=0.18, zorder=2)
    for vl in run_vals:
        ax.plot(vl, color=col, linewidth=0.6, alpha=0.18, linestyle='--', zorder=2)

    # ── Mean across runs ─────────────────────────────────────────────────
    if run_trains:
        max_len = max(len(c) for c in run_trains)
        arr_t   = np.full((len(run_trains), max_len), np.nan)
        arr_v   = np.full((len(run_vals),   max_len), np.nan)
        for ri, (tl, vl) in enumerate(zip(run_trains, run_vals)):
            arr_t[ri, :len(tl)] = tl
            arr_v[ri, :len(vl)] = vl
        mean_t = np.nanmean(arr_t, axis=0)
        mean_v = np.nanmean(arr_v, axis=0)
        ax.plot(mean_t, color=col, linewidth=1.8, alpha=0.65, zorder=3)
        ax.plot(mean_v, color=col, linewidth=1.8, alpha=0.65, linestyle='--', zorder=3)

    # ── Best model (bold) ─────────────────────────────────────────────────
    if best_tl:
        ax.plot(best_tl, color=col, linewidth=2.5, alpha=1.0, zorder=5,
                label='Train (best)')
    if best_vl:
        ax.plot(best_vl, color=col, linewidth=2.5, alpha=1.0, zorder=5,
                linestyle='--', label='Val (best)')

    ax.set_xlabel('Epoch', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('Loss',  fontsize=FONT_SIZES['label'])
    ax.set_title(meta['label'], fontsize=FONT_SIZES['subplot_title'], pad=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=FONT_SIZES['tick'])

    ax.text(-0.10, 1.06, PANEL_LABELS[pi], transform=ax.transAxes,
            fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='bottom')

# ── Shared legend ─────────────────────────────────────────────────────────
train_line = mlines.Line2D([], [], color='0.3', linewidth=2.0,
                           linestyle='-',  label='Train')
val_line   = mlines.Line2D([], [], color='0.3', linewidth=2.0,
                           linestyle='--', label='Validation')
best_line  = mlines.Line2D([], [], color='0.3', linewidth=2.5,
                           linestyle='-',  alpha=1.0, label='Best model (bold)')
run_line   = mlines.Line2D([], [], color='0.3', linewidth=0.8,
                           linestyle='-',  alpha=0.4, label='50-run mean / runs')

fig.legend(handles=[train_line, val_line, best_line, run_line],
           loc='lower center', ncol=4, fontsize=FONT_SIZES['small'],
           frameon=False, bbox_to_anchor=(0.5, 0.00))

# ── Save ──────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
stem = os.path.join(OUT_DIR, 'figS_training_curves')
fig.savefig(stem + '.pdf', dpi=300, bbox_inches='tight')
fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {stem}.pdf / .png')
