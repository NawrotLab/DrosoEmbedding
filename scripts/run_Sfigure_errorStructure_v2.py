"""
Supplementary Figure — Error Structure Analysis v2 (16-class task)
===================================================================

Extends the original 4-type error scheme to 7 types by decomposing
the former monolithic "cross-modality" bin into four sub-types depending
on whether state and/or valence also changed.  This allows the manuscript
to distinguish *cross-hierarchy* errors (Types 5 and 7) — those that cross
both the modality AND the state boundary — from simpler cross-modality
confusions.

New 7-type taxonomy
-------------------
Single-factor
  1  Valence only      same modality, same state,  different valence
  2  State only        same modality, different state, same valence
  3  Modality only     different modality, same state,  same valence
Two-factor
  4  State × Valence   same modality, both differ
  5  State × Modality  different modality + different state, same valence  [cross-hierarchy]
  6  Modality × Valence  different modality + different valence, same state
Three-factor
  7  All three         all differ                                          [cross-hierarchy]

Panel A  Seven error-type bars with 50-run scatter overlay
Panel B  Per-modality error-rate split (within / cross)
Panel C  3×3 modality-level confusion heatmap (row-normalised)

Usage (from repo root):
    python scripts/run_Sfigure_errorStructure_v2.py

Output:
    results/CombiPlots/figS_errorStructure_v2.pdf
    results/CombiPlots/figS_errorStructure_v2.png
"""

import gc
import os
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
TASK_KEY         = 'State_Modality_Valence_16'

CLASS_NAMES = [
    r'O$^{+}$ (S)',          r'O$^{-}$ (S)',          r'O$^{+}$ (F)',          r'O$^{-}$ (F)',
    r'T$^{+}$ (S)',          r'T$^{-}$ (S)',          r'T$^{+}$ (F)',          r'T$^{-}$ (F)',
    r'O$^{+}$+T$^{+}$ (S)', r'O$^{-}$+T$^{-}$ (S)',
    r'O$^{-}$+T$^{+}$ (S)', r'O$^{+}$+T$^{-}$ (S)',
    r'O$^{+}$+T$^{+}$ (F)', r'O$^{-}$+T$^{-}$ (F)',
    r'O$^{-}$+T$^{+}$ (F)', r'O$^{+}$+T$^{-}$ (F)',
]

# (modality, state, valence)
# Valence: int for Odor/Taste, tuple (odor_val, taste_val) for Combined
CLASS_PROPS = [
    ('Odor',     'Starved', +1),        # 0
    ('Odor',     'Starved', -1),        # 1
    ('Odor',     'Fed',     +1),        # 2
    ('Odor',     'Fed',     -1),        # 3
    ('Taste',    'Starved', +1),        # 4
    ('Taste',    'Starved', -1),        # 5
    ('Taste',    'Fed',     +1),        # 6
    ('Taste',    'Fed',     -1),        # 7
    ('Combined', 'Starved', (+1, +1)),  # 8
    ('Combined', 'Starved', (-1, -1)),  # 9
    ('Combined', 'Starved', (-1, +1)),  # 10  conflict
    ('Combined', 'Starved', (+1, -1)),  # 11  conflict
    ('Combined', 'Fed',     (+1, +1)),  # 12
    ('Combined', 'Fed',     (-1, -1)),  # 13
    ('Combined', 'Fed',     (-1, +1)),  # 14  conflict
    ('Combined', 'Fed',     (+1, -1)),  # 15  conflict
]

MODALITY_GROUPS = {
    'Odor':     list(range(0, 4)),
    'Taste':    list(range(4, 8)),
    'Combined': list(range(8, 16)),
}
MODALITY_ORDER   = ['Odor', 'Taste', 'Combined']
MODALITY_COLOURS = {
    'Odor':     '#D35F2A',
    'Taste':    '#3382BE',
    'Combined': '#7B4278',
}

# 7-type Okabe-Ito-extended colorblind-safe palette
ERROR_TYPE_COLOURS = {
    1: '#E69F00',   # orange     — Valence only (dominant)
    2: '#0072B2',   # blue       — State only
    3: '#009E73',   # green      — Modality only
    4: '#CC79A7',   # pink       — State × Valence
    5: '#D55E00',   # vermillion — State × Modality  [cross-hierarchy]
    6: '#56B4E9',   # sky blue   — Modality × Valence
    7: '#999999',   # grey       — All three          [cross-hierarchy]
}
ERROR_TYPE_LABELS = {
    1: 'Type 1\nValence\nonly',
    2: 'Type 2\nState\nonly',
    3: 'Type 3\nModality\nonly',
    4: 'Type 4\nState ×\nValence',
    5: 'Type 5\nState ×\nModality',
    6: 'Type 6\nModality ×\nValence',
    7: 'Type 7\nAll three',
}

# ════════════════════════════════════════════════
# ERROR CLASSIFICATION
# ════════════════════════════════════════════════

def net_valence(cls_idx):
    """Net valence: +1 / -1 for congruent/single; 0 for conflict (ambiguous)."""
    _, _, val = CLASS_PROPS[cls_idx]
    if isinstance(val, tuple):
        return val[0] if val[0] == val[1] else 0
    return val


def classify_error_7(i, j):
    """Return error type 1-7 for misclassification from true class i to predicted j."""
    mod_i, state_i, _ = CLASS_PROPS[i]
    mod_j, state_j, _ = CLASS_PROPS[j]

    same_mod   = (mod_i == mod_j)
    same_state = (state_i == state_j)

    nv_i = net_valence(i)
    nv_j = net_valence(j)
    # "same valence" only when both non-conflict and equal
    same_val = (nv_i != 0 and nv_j != 0 and nv_i == nv_j)

    if same_mod:
        # Within-modality
        if same_state:
            return 1                   # Valence only (state same → val must differ for i≠j)
        if same_val:
            return 2                   # State only
        return 4                       # State × Valence
    else:
        # Cross-modality
        if same_state and same_val:
            return 3                   # Modality only
        if not same_state and same_val:
            return 5                   # State × Modality  [cross-hierarchy]
        if same_state and not same_val:
            return 6                   # Modality × Valence
        return 7                       # All three          [cross-hierarchy]


def error_type_pct(cm):
    """Compute % of total errors for each of the 7 types from a 16×16 confusion matrix."""
    counts = {t: 0 for t in range(1, 8)}
    total  = 0
    for i in range(16):
        for j in range(16):
            if i == j:
                continue
            n = int(cm[i, j])
            if n == 0:
                continue
            counts[classify_error_7(i, j)] += n
            total += n
    if total == 0:
        return {t: 0.0 for t in range(1, 8)}
    return {t: 100.0 * v / total for t, v in counts.items()}


# ════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════

print(f'Loading results from {BASE_RESULTS_DIR} …')
results_dict = load_all_results(BASE_RESULTS_DIR, task_names=[TASK_KEY])

if TASK_KEY not in results_dict or results_dict[TASK_KEY].get('best') is None:
    raise FileNotFoundError(
        f"No 'best' result found for '{TASK_KEY}' in {BASE_RESULTS_DIR}.\n"
        "Run run_evaluation.py first."
    )

entry  = results_dict[TASK_KEY]
cm_raw = entry['best']['confusion_matrix']
assert cm_raw.shape == (16, 16), f"Expected 16×16 CM, got {cm_raw.shape}"

# ── 50-run confusion matrices ──────────────────────────────────────────────
all_cms = []
h16_runs = entry.get('runs', {}).get('H16', [])
print(f'Loading confusion matrices from {len(h16_runs)} H16 runs …')
for run in h16_runs:
    path = run.get('path')
    if path is None:
        continue
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        cm = data.get('confusion_matrix')
        if cm is not None and np.array(cm).shape == (16, 16):
            all_cms.append(np.array(cm))
        del data
        gc.collect()
    except Exception as e:
        print(f'  [WARN] {path}: {e}')
print(f'Loaded {len(all_cms)} run confusion matrices.')

# ════════════════════════════════════════════════
# COMPUTE ERROR STATISTICS
# ════════════════════════════════════════════════

# Best model
best_pct = error_type_pct(cm_raw)

# 50 runs
run_pcts = np.array([[error_type_pct(cm)[t] for t in range(1, 8)] for cm in all_cms])  # (n_runs, 7)
run_mean = run_pcts.mean(axis=0) if len(run_pcts) > 0 else np.zeros(7)

# Per-modality breakdown (unchanged)
modality_stats = {}
for mod in MODALITY_ORDER:
    idx       = np.array(MODALITY_GROUPS[mod])
    N_total   = int(cm_raw[idx, :].sum())
    N_correct = int(cm_raw[np.ix_(idx, idx)].diagonal().sum())
    N_errors  = N_total - N_correct
    within_block = cm_raw[np.ix_(idx, idx)]
    N_within_err = int(within_block.sum()) - N_correct
    N_cross_err  = N_errors - N_within_err
    error_rate   = 100.0 * N_errors    / N_total  if N_total  > 0 else 0.0
    within_pct   = 100.0 * N_within_err / N_errors if N_errors > 0 else 0.0
    cross_pct    = 100.0 * N_cross_err  / N_errors if N_errors > 0 else 0.0
    modality_stats[mod] = dict(
        N_total=N_total, N_correct=N_correct, N_errors=N_errors,
        error_rate=error_rate, within_pct=within_pct, cross_pct=cross_pct,
        within_abs=error_rate * within_pct / 100,
        cross_abs =error_rate * cross_pct  / 100,
    )

# 3×3 modality confusion
mod_cm = np.zeros((3, 3))
for i, true_mod in enumerate(MODALITY_ORDER):
    true_idx = np.array(MODALITY_GROUPS[true_mod])
    N_true   = int(cm_raw[true_idx, :].sum())
    for j, pred_mod in enumerate(MODALITY_ORDER):
        pred_idx     = np.array(MODALITY_GROUPS[pred_mod])
        count        = int(cm_raw[np.ix_(true_idx, pred_idx)].sum())
        mod_cm[i, j] = 100.0 * count / N_true if N_true > 0 else 0.0

# ════════════════════════════════════════════════
# PRINT SUMMARY
# ════════════════════════════════════════════════

total_errors = sum(int(cm_raw[i, j]) for i in range(16) for j in range(16) if i != j)
cross_hier   = best_pct[5] + best_pct[7]

print('\n' + '=' * 65)
print('ERROR STRUCTURE SUMMARY  (7-type taxonomy)')
print('=' * 65)
print(f'Total misclassifications (best run): {total_errors}')
print()
for t in range(1, 8):
    tag   = ' ← cross-hierarchy' if t in (5, 7) else ''
    label = ERROR_TYPE_LABELS[t].replace('\n', ' ')
    print(f'  Type {t}  {label:30s}: {best_pct[t]:5.1f}%{tag}')
print(f'\n  Cross-hierarchy total (Types 5+7): {cross_hier:.1f}%')
if len(run_pcts) > 0:
    print(f'\n  50-run means:')
    for t in range(1, 8):
        print(f'    Type {t}: {run_mean[t-1]:.1f}%  '
              f'(IQR {np.percentile(run_pcts[:, t-1], 25):.1f}–{np.percentile(run_pcts[:, t-1], 75):.1f}%)')
print()
print('Per-modality breakdown:')
for mod in MODALITY_ORDER:
    s = modality_stats[mod]
    print(f'  {mod:10s}: {s["error_rate"]:5.1f}% error rate'
          f'  (within {s["within_pct"]:4.1f}%  /  cross {s["cross_pct"]:4.1f}%)')
print('=' * 65)

# ════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════

fig = plt.figure(figsize=(14, 5))
gs  = GridSpec(1, 3, figure=fig,
               left=0.06, right=0.97, top=0.88, bottom=0.25,
               wspace=0.40, width_ratios=[2.2, 1.0, 0.95])
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])
ax_c = fig.add_subplot(gs[2])

rng = np.random.default_rng(42)

# ── Panel A: 7-type error hierarchy ───────────────────────────────────────

xs      = np.arange(1, 8)
heights = [best_pct[t] for t in range(1, 8)]
colours = [ERROR_TYPE_COLOURS[t] for t in range(1, 8)]
max_h   = max(heights)

bars_a = ax_a.bar(xs, heights, width=0.55, color=colours, alpha=0.75,
                  edgecolor='none', zorder=2)

# 50-run scatter overlay
if len(run_pcts) > 0:
    for ti, t in enumerate(range(1, 8)):
        ys     = run_pcts[:, ti]
        jitter = rng.uniform(-0.18, 0.18, size=len(ys))
        ax_a.scatter(
            np.full(len(ys), t) + jitter, ys,
            s=14, color=ERROR_TYPE_COLOURS[t],
            alpha=0.35, linewidths=0, zorder=4,
        )
        # 50-run mean as a short black horizontal segment
        ax_a.plot(
            [t - 0.22, t + 0.22], [run_mean[ti], run_mean[ti]],
            color='black', linewidth=1.5, zorder=5,
        )

# Best-model value labels above bars
for bar, h in zip(bars_a, heights):
    ax_a.text(
        bar.get_x() + bar.get_width() / 2,
        h + max_h * 0.025,
        f'{h:.1f}%',
        ha='center', va='bottom',
        fontsize=FONT_SIZES['small'], fontweight='bold',
    )

# Cross-hierarchy bracket annotation (Types 5 and 7)
y_bracket = max_h * 1.12
ax_a.annotate(
    '',
    xy=(5, y_bracket), xytext=(7, y_bracket),
    arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.2),
)
ax_a.text(6, y_bracket + max_h * 0.03,
          f'Cross-hierarchy\n(Types 5+7): {cross_hier:.1f}%',
          ha='center', va='bottom',
          fontsize=FONT_SIZES['small'] - 1, color='#555555', style='italic')

ax_a.set_xticks(xs)
ax_a.set_xticklabels([ERROR_TYPE_LABELS[t] for t in range(1, 8)],
                     fontsize=FONT_SIZES['small'])
ax_a.set_ylabel('% of total errors', fontsize=FONT_SIZES['label'])
ax_a.set_ylim(0, max_h * 1.50)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.tick_params(axis='x', length=0)

# 50-run mean legend entry
if len(run_pcts) > 0:
    mean_line = plt.Line2D([0], [0], color='black', linewidth=1.5, label='50-run mean')
    ax_a.legend(handles=[mean_line], fontsize=FONT_SIZES['small'] - 1,
                frameon=False, loc='upper left')

ax_a.text(-0.08, 1.04, 'A', transform=ax_a.transAxes,
          fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='bottom')

# ── Panel B: Per-modality error rate (within / cross stacked) ─────────────

x_b   = np.arange(len(MODALITY_ORDER))
bar_w = 0.52

for i, mod in enumerate(MODALITY_ORDER):
    s   = modality_stats[mod]
    col = MODALITY_COLOURS[mod]
    ax_b.bar(x_b[i], s['within_abs'], width=bar_w, color=col, alpha=0.65, zorder=1)
    ax_b.bar(x_b[i], s['cross_abs'], width=bar_w, bottom=s['within_abs'],
             color=col, alpha=0.25, zorder=1)
    ax_b.bar(x_b[i], s['error_rate'], width=bar_w,
             facecolor='none', edgecolor=col, linewidth=2, zorder=2)
    top = max(modality_stats[m]['error_rate'] for m in MODALITY_ORDER)
    ax_b.text(x_b[i], s['error_rate'] + top * 0.025,
              f"{s['error_rate']:.1f}%",
              ha='center', va='bottom',
              fontsize=FONT_SIZES['small'], fontweight='bold', color=col)

ax_b.set_xticks(x_b)
ax_b.set_xticklabels(MODALITY_ORDER, fontsize=FONT_SIZES['tick'])
ax_b.set_ylabel('Error rate (% of modality samples)', fontsize=FONT_SIZES['label'])
ax_b.set_ylim(0, max(s['error_rate'] for s in modality_stats.values()) * 1.30)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.tick_params(axis='x', length=0)

within_patch = mpatches.Patch(color='#888888', alpha=0.65, label='Within-modality')
cross_patch  = mpatches.Patch(color='#888888', alpha=0.25, label='Cross-modality')
ax_b.legend(handles=[within_patch, cross_patch],
            fontsize=FONT_SIZES['small'] - 1, frameon=False, loc='upper left')

ax_b.text(-0.22, 1.04, 'B', transform=ax_b.transAxes,
          fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='bottom')

# ── Panel C: 3×3 modality confusion heatmap ───────────────────────────────

ax_c.imshow(mod_cm, vmin=0, vmax=100, cmap='Blues', aspect='auto')
for i in range(3):
    for j in range(3):
        v       = mod_cm[i, j]
        txt_col = 'white' if v > 58 else '#222222'
        ax_c.text(j, i, f'{v:.1f}%',
                  ha='center', va='center',
                  fontsize=FONT_SIZES['heatmap_cell'],
                  color=txt_col,
                  fontweight='bold' if i == j else 'normal')

ax_c.set_xticks(range(3))
ax_c.set_yticks(range(3))
ax_c.set_xticklabels(MODALITY_ORDER, fontsize=FONT_SIZES['tick'])
ax_c.set_yticklabels(MODALITY_ORDER, fontsize=FONT_SIZES['tick'])
ax_c.set_xlabel('Predicted modality', fontsize=FONT_SIZES['label'])
ax_c.set_ylabel('True modality',      fontsize=FONT_SIZES['label'])
ax_c.set_title('Modality-level confusion\n(row-normalised)',
               fontsize=FONT_SIZES['subplot_title'], pad=8)

for tick, mod in zip(ax_c.get_xticklabels(), MODALITY_ORDER):
    tick.set_color(MODALITY_COLOURS[mod])
for tick, mod in zip(ax_c.get_yticklabels(), MODALITY_ORDER):
    tick.set_color(MODALITY_COLOURS[mod])

ax_c.text(-0.34, 1.04, 'C', transform=ax_c.transAxes,
          fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='bottom')

# ════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════

os.makedirs(OUT_DIR, exist_ok=True)
stem = os.path.join(OUT_DIR, 'figS_errorStructure_v2')
fig.savefig(stem + '.pdf', dpi=300, bbox_inches='tight')
fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {stem}.pdf / .png')
