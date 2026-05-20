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

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

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
    tl   = d.get('train')
    vl   = d.get('val')

    if tl:
        ax.plot(tl, color=col, linewidth=2.0, label='Train')
    if vl:
        ax.plot(vl, color=col, linewidth=2.0, linestyle='--', label='Validation')

    ax.set_xlabel('Epoch', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('Loss',  fontsize=FONT_SIZES['label'])
    ax.set_title(meta['label'], fontsize=FONT_SIZES['subplot_title'], pad=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=FONT_SIZES['tick'])
    ax.text(-0.10, 1.06, PANEL_LABELS[pi], transform=ax.transAxes,
            fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='bottom')

# ── Shared legend ─────────────────────────────────────────────────────────
train_line = mlines.Line2D([], [], color='0.4', linewidth=2.0,
                           linestyle='-',  label='Training loss')
val_line   = mlines.Line2D([], [], color='0.4', linewidth=2.0,
                           linestyle='--', label='Validation loss')
fig.legend(handles=[train_line, val_line],
           loc='lower center', ncol=2, fontsize=FONT_SIZES['legend'],
           frameon=False, bbox_to_anchor=(0.5, 0.00))

# ── Save ──────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
stem = os.path.join(OUT_DIR, 'figS_training_curves')
fig.savefig(stem + '.pdf', dpi=300, bbox_inches='tight')
fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {stem}.pdf / .png')
