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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization.figure_base import apply_style, FONT_SIZES

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
    'MetabolicState_2':          {'label': 'i.  State (2-class)',                        'color': '#0072B2'},
    'State_Modality_6':          {'label': 'ii.  State × Modality (6-class)',             'color': '#D55E00'},
    'State_Modality_Valence_16': {'label': 'iii.  State × Modality × Valence (16-class)', 'color': '#009E73'},
}

# ════════════════════════════════════════════════
# LOAD DATA  (best pkl only — one file per task)
# ════════════════════════════════════════════════

def _find_latest_pkl(directory):
    d = Path(directory)
    if not d.is_dir():
        return None
    pkls = sorted(d.glob('*.pkl'), key=lambda p: p.stat().st_mtime, reverse=True)
    return pkls[0] if pkls else None

def _load_losses(path):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        tl = list(data.get('train_loss') or [])
        vl = list(data.get('val_loss')   or [])
        del data
        gc.collect()
        return (tl, vl) if tl and vl else (None, None)
    except Exception as e:
        print(f'  [WARN] {path}: {e}')
        return None, None

task_data = {}
for task_key in TASK_ORDER:
    meta      = TASK_META[task_key]
    best_path = _find_latest_pkl(Path(BASE_RESULTS_DIR) / task_key / 'best')
    if best_path is None:
        print(f'[WARN] No best pkl found for {task_key}')
        task_data[task_key] = {'train': None, 'val': None}
        continue
    print(f'Loading {task_key}: {best_path.name} …')
    tl, vl = _load_losses(best_path)
    task_data[task_key] = {'train': tl, 'val': vl}
    if tl:
        print(f'  → {len(tl)} epochs')

# ════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════

SMOOTH_WIN = 7   # moving-average window (epochs)

def _smooth(values, window):
    if not values or window < 2:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid').tolist()


fig, ax = plt.subplots(figsize=(8, 5))
fig.subplots_adjust(left=0.11, right=0.97, top=0.93, bottom=0.22)

legend_handles = []

for task_key in TASK_ORDER:
    meta = TASK_META[task_key]
    col  = meta['color']
    d    = task_data.get(task_key, {})
    tl   = _smooth(d.get('train'), SMOOTH_WIN)
    vl   = _smooth(d.get('val'),   SMOOTH_WIN)

    if tl:
        ax.plot(tl, color=col, linewidth=2.0, alpha=1.0)
    if vl:
        ax.plot(vl, color=col, linewidth=2.0, alpha=0.4)

    legend_handles.append(
        mlines.Line2D([], [], color=col, linewidth=2.0, label=meta['label'])
    )

# Style entries for train / val
legend_handles += [
    mlines.Line2D([], [], color='0.4', linewidth=1.5, alpha=1.0, label='Training'),
    mlines.Line2D([], [], color='0.4', linewidth=1.5, alpha=0.4, label='Validation'),
]

ax.set_xlabel('Epoch', fontsize=FONT_SIZES['label'])
ax.set_ylabel('Loss',  fontsize=FONT_SIZES['label'])
ax.set_xlim(0, 800)
ax.set_ylim(0, 2.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_bounds(0, 2.5)
ax.spines['bottom'].set_bounds(0, 800)
ax.tick_params(axis='both', labelsize=FONT_SIZES['tick'])

ax.legend(handles=legend_handles, fontsize=FONT_SIZES['legend'],
          frameon=False, loc='upper right', ncol=1)

# ── Save ──────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
stem = os.path.join(OUT_DIR, 'figS_training_curves')
fig.savefig(stem + '.pdf', dpi=300, bbox_inches='tight')
fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {stem}.pdf / .png')
