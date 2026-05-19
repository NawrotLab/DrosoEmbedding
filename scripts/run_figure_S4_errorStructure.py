"""
Supplementary Figure S4 — Error Structure Analysis (16-class task)
===================================================================

Analyses and visualises the hierarchical error structure of the 16-class
(State × Modality × Valence) classifier.  Three panels make key manuscript
claims directly verifiable:

  Panel A  Error-type hierarchy — valence errors dominate; cross-modality ~0%
  Panel B  Per-modality error rate split into within vs. cross-modality portion
  Panel C  Modality-level confusion (3×3 row-normalised matrix)

Manuscript claims verified:
  "Dominant confusion axis was valence"             → Type 1 bar in Panel A
  "Secondary confusion axis was metabolic state"    → Type 2 bar in Panel A
  "Cross-hierarchy errors were virtually absent"    → Type 4 bar in Panel A, Panel C
  "Combined conditions showed most internal confusion" → Panel B / Panel C

Usage (from repo root):
    python scripts/run_figure_S4_errorStructure.py

Output:
    /mnt/user-data/outputs/figS4_errorStructure.pdf
    /mnt/user-data/outputs/figS4_errorStructure.png
"""

import os
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
OUT_DIR = os.path.join('results', 'CombiPlots')
TASK_KEY = 'State_Modality_Valence_16'

# 16 class names in label order, matching classification_report_dict keys
CLASS_NAMES = [
    r'O$^{+}$ (S)',          r'O$^{-}$ (S)',          r'O$^{+}$ (F)',          r'O$^{-}$ (F)',          # 0-3  Odour
    r'T$^{+}$ (S)',          r'T$^{-}$ (S)',          r'T$^{+}$ (F)',          r'T$^{-}$ (F)',          # 4-7  Taste
    r'O$^{+}$+T$^{+}$ (S)', r'O$^{-}$+T$^{-}$ (S)',                                                    # 8-9
    r'O$^{-}$+T$^{+}$ (S)', r'O$^{+}$+T$^{-}$ (S)',                                                    # 10-11
    r'O$^{+}$+T$^{+}$ (F)', r'O$^{-}$+T$^{-}$ (F)',                                                    # 12-13
    r'O$^{-}$+T$^{+}$ (F)', r'O$^{+}$+T$^{-}$ (F)',                                                    # 14-15 Combined
]

# Per-class properties: (modality, metabolic_state, valence)
# Valence encoding: for Odour/Taste classes, int (+1=appetitive, -1=aversive)
#                  for Combined classes, tuple (odour_val, taste_val)
CLASS_PROPS = [
    ('Odour',    'Starved', +1),         # 0  O+ (S)
    ('Odour',    'Starved', -1),         # 1  O- (S)
    ('Odour',    'Fed',     +1),         # 2  O+ (F)
    ('Odour',    'Fed',     -1),         # 3  O- (F)
    ('Taste',    'Starved', +1),         # 4  T+ (S)
    ('Taste',    'Starved', -1),         # 5  T- (S)
    ('Taste',    'Fed',     +1),         # 6  T+ (F)
    ('Taste',    'Fed',     -1),         # 7  T- (F)
    ('Combined', 'Starved', (+1, +1)),   # 8  O+T+ (S)  congruent appetitive
    ('Combined', 'Starved', (-1, -1)),   # 9  O-T- (S)  congruent aversive
    ('Combined', 'Starved', (-1, +1)),   # 10 O-T+ (S)  conflict
    ('Combined', 'Starved', (+1, -1)),   # 11 O+T- (S)  conflict
    ('Combined', 'Fed',     (+1, +1)),   # 12 O+T+ (F)
    ('Combined', 'Fed',     (-1, -1)),   # 13 O-T- (F)
    ('Combined', 'Fed',     (-1, +1)),   # 14 O-T+ (F)
    ('Combined', 'Fed',     (+1, -1)),   # 15 O+T- (F)
]

MODALITY_GROUPS = {
    'Odour':    list(range(0, 4)),
    'Taste':    list(range(4, 8)),
    'Combined': list(range(8, 16)),
}
MODALITY_ORDER = ['Odour', 'Taste', 'Combined']
MODALITY_COLOURS = {
    'Odour':    '#D35F2A',
    'Taste':    '#3382BE',
    'Combined': '#7B4278',
}

# Okabe-Ito colorblind-safe palette for error types
ERROR_TYPE_COLOURS = {
    1: '#E69F00',   # orange   — Type 1 valence flip (dominant)
    2: '#0072B2',   # blue     — Type 2 state flip (secondary)
    3: '#009E73',   # green    — Type 3 both flipped
    4: '#999999',   # grey     — Type 4 cross-modality (negligible)
}
ERROR_TYPE_LABELS = {
    1: 'Type 1\nValence only',
    2: 'Type 2\nState only',
    3: 'Type 3\nValence\n+ State',
    4: 'Type 4\nCross-\nmodality',
}

# ════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════

print(f'Loading results from {BASE_RESULTS_DIR} …')
results_dict = load_all_results(BASE_RESULTS_DIR, task_names=[TASK_KEY])

if TASK_KEY not in results_dict or results_dict[TASK_KEY].get('best') is None:
    raise FileNotFoundError(
        f"No 'best' result found for task '{TASK_KEY}' in {BASE_RESULTS_DIR}.\n"
        "Run run_evaluation.py first to generate evaluation results."
    )

cm_raw = results_dict[TASK_KEY]['best']['confusion_matrix']   # 16×16 absolute counts
assert cm_raw.shape == (16, 16), f"Expected 16×16 confusion matrix, got {cm_raw.shape}"

# ════════════════════════════════════════════════
# ERROR TYPE CLASSIFICATION
# ════════════════════════════════════════════════

def classify_error(i, j):
    """Return error type (1-4) for a misclassification from true class i to predicted j."""
    mod_i, state_i, val_i = CLASS_PROPS[i]
    mod_j, state_j, val_j = CLASS_PROPS[j]

    if mod_i != mod_j:
        return 4                          # different modality → cross-hierarchy

    same_state = (state_i == state_j)
    same_val   = (val_i == val_j)        # int==int or tuple==tuple (never mixed)

    if same_state and not same_val:
        return 1                          # valence flip only
    if not same_state and same_val:
        return 2                          # state flip only
    return 3                              # both state and valence differ


type_counts = {1: 0, 2: 0, 3: 0, 4: 0}
total_errors = 0

for i in range(16):
    for j in range(16):
        if i == j:
            continue                      # diagonal = correct classifications
        n = int(cm_raw[i, j])
        if n == 0:
            continue
        type_counts[classify_error(i, j)] += n
        total_errors += n

type_pct = {t: 100.0 * v / total_errors for t, v in type_counts.items()}

# ════════════════════════════════════════════════
# PER-MODALITY ERROR DECOMPOSITION
# ════════════════════════════════════════════════

modality_stats = {}
for mod in MODALITY_ORDER:
    idx       = np.array(MODALITY_GROUPS[mod])
    N_total   = int(cm_raw[idx, :].sum())
    N_correct = int(cm_raw[np.ix_(idx, idx)].diagonal().sum())
    N_errors  = N_total - N_correct

    # Within-modality errors: misclassified but stayed inside the modality block
    within_block  = cm_raw[np.ix_(idx, idx)]
    N_within_err  = int(within_block.sum()) - N_correct
    N_cross_err   = N_errors - N_within_err

    error_rate  = 100.0 * N_errors    / N_total  if N_total  > 0 else 0.0
    within_pct  = 100.0 * N_within_err / N_errors if N_errors > 0 else 0.0
    cross_pct   = 100.0 * N_cross_err  / N_errors if N_errors > 0 else 0.0

    modality_stats[mod] = dict(
        N_total=N_total, N_correct=N_correct, N_errors=N_errors,
        N_within_err=N_within_err, N_cross_err=N_cross_err,
        error_rate=error_rate, within_pct=within_pct, cross_pct=cross_pct,
        within_abs=error_rate * within_pct / 100,   # absolute % of total samples
        cross_abs =error_rate * cross_pct  / 100,
    )

# ════════════════════════════════════════════════
# 3×3 MODALITY-LEVEL CONFUSION (row-normalised)
# ════════════════════════════════════════════════

mod_cm = np.zeros((3, 3))
for i, true_mod in enumerate(MODALITY_ORDER):
    true_idx = np.array(MODALITY_GROUPS[true_mod])
    N_true   = int(cm_raw[true_idx, :].sum())
    for j, pred_mod in enumerate(MODALITY_ORDER):
        pred_idx    = np.array(MODALITY_GROUPS[pred_mod])
        count       = int(cm_raw[np.ix_(true_idx, pred_idx)].sum())
        mod_cm[i, j] = 100.0 * count / N_true if N_true > 0 else 0.0

# ════════════════════════════════════════════════
# PRINT SUMMARY STATISTICS
# ════════════════════════════════════════════════

print('\n' + '=' * 60)
print('ERROR STRUCTURE SUMMARY')
print('=' * 60)
print(f'Total misclassifications: {total_errors}')
print()
for t in [1, 2, 3, 4]:
    desc = ERROR_TYPE_LABELS[t].replace('\n', ' ')
    print(f'  Type {t} ({desc:25s}): {type_counts[t]:5d}  ({type_pct[t]:5.1f}%)')
print()
print('Per-modality breakdown:')
for mod in MODALITY_ORDER:
    s = modality_stats[mod]
    print(f'  {mod:10s}: {s["error_rate"]:5.1f}% error rate'
          f'  (within-modality {s["within_pct"]:4.1f}%'
          f'  /  cross-modality {s["cross_pct"]:4.1f}%)')
print()
print('3×3 Modality confusion (row-normalised %):')
header = ''.join(f'{"→ " + m:>15s}' for m in MODALITY_ORDER)
print(f'{"":12s}{header}')
for i, mod in enumerate(MODALITY_ORDER):
    row = ''.join(f'{v:13.1f}%  ' for v in mod_cm[i])
    print(f'  {mod:10s}  {row}')
print('=' * 60)

# ════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════

fig = plt.figure(figsize=(12, 4.8))
gs  = GridSpec(1, 3, figure=fig,
               left=0.07, right=0.97, top=0.87, bottom=0.20,
               wspace=0.42, width_ratios=[1.3, 1.0, 0.95])
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])
ax_c = fig.add_subplot(gs[2])

# ── Panel A: Error-type hierarchy ─────────────────────────────

xs      = np.arange(1, 5)
heights = [type_pct[t] for t in [1, 2, 3, 4]]
colours = [ERROR_TYPE_COLOURS[t] for t in [1, 2, 3, 4]]

bars_a = ax_a.bar(xs, heights, width=0.62, color=colours, edgecolor='none', zorder=2)

for bar, h in zip(bars_a, heights):
    label_y = h + max(heights) * 0.02
    ax_a.text(bar.get_x() + bar.get_width() / 2, label_y,
              f'{h:.1f}%', ha='center', va='bottom',
              fontsize=FONT_SIZES['small'], fontweight='bold')

# Extra call-out for the negligible cross-modality bar
ax_a.annotate(
    f'Cross-modality:\n{type_pct[4]:.1f}%',
    xy=(4, type_pct[4]),
    xytext=(3.45, max(heights) * 0.55),
    fontsize=FONT_SIZES['small'] - 1,
    color='#555555',
    style='italic',
    arrowprops=dict(arrowstyle='->', color='#888888', lw=1.0),
    ha='right',
)

ax_a.set_xticks(xs)
ax_a.set_xticklabels([ERROR_TYPE_LABELS[t] for t in [1, 2, 3, 4]],
                     fontsize=FONT_SIZES['small'])
ax_a.set_ylabel('% of total errors', fontsize=FONT_SIZES['label'])
ax_a.set_ylim(0, max(heights) * 1.30)
ax_a.set_title('Error-type hierarchy', fontsize=FONT_SIZES['subplot_title'], pad=8)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.tick_params(axis='x', length=0)

ax_a.text(-0.10, 1.04, 'A', transform=ax_a.transAxes,
          fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='bottom')

# ── Panel B: Per-modality error rate (outline = total; fill = within) ──

x_b   = np.arange(len(MODALITY_ORDER))
bar_w = 0.52

for i, mod in enumerate(MODALITY_ORDER):
    s   = modality_stats[mod]
    col = MODALITY_COLOURS[mod]

    # Filled bar: within-modality error rate (absolute, % of all samples in mod)
    ax_b.bar(x_b[i], s['within_abs'], width=bar_w,
             color=col, alpha=0.65, zorder=1)

    # Filled bar stacked: cross-modality error rate (lighter)
    ax_b.bar(x_b[i], s['cross_abs'], width=bar_w, bottom=s['within_abs'],
             color=col, alpha=0.25, zorder=1)

    # Outline bar at total height (drawn last so border is crisp)
    ax_b.bar(x_b[i], s['error_rate'], width=bar_w,
             facecolor='none', edgecolor=col, linewidth=2, zorder=2)

    # Total error rate above bar
    ax_b.text(x_b[i], s['error_rate'] + max(modality_stats[m]['error_rate'] for m in MODALITY_ORDER) * 0.025,
              f"{s['error_rate']:.1f}%",
              ha='center', va='bottom',
              fontsize=FONT_SIZES['small'], fontweight='bold', color=col)

ax_b.set_xticks(x_b)
ax_b.set_xticklabels(MODALITY_ORDER, fontsize=FONT_SIZES['tick'])
ax_b.set_ylabel('Error rate (% of modality samples)', fontsize=FONT_SIZES['label'])
ax_b.set_ylim(0, max(s['error_rate'] for s in modality_stats.values()) * 1.30)
ax_b.set_title('Per-modality error rate', fontsize=FONT_SIZES['subplot_title'], pad=8)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.tick_params(axis='x', length=0)

within_patch = mpatches.Patch(color='#888888', alpha=0.65, label='Within-modality errors')
cross_patch  = mpatches.Patch(color='#888888', alpha=0.25, label='Cross-modality errors')
ax_b.legend(handles=[within_patch, cross_patch],
            fontsize=FONT_SIZES['small'] - 1, frameon=False, loc='upper left')

ax_b.text(-0.20, 1.04, 'B', transform=ax_b.transAxes,
          fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='bottom')

# ── Panel C: 3×3 modality confusion heatmap ────────────────────

ax_c.imshow(mod_cm, vmin=0, vmax=100, cmap='Blues', aspect='auto')

for i in range(3):
    for j in range(3):
        v        = mod_cm[i, j]
        is_diag  = (i == j)
        txt_col  = 'white' if v > 58 else '#222222'
        ax_c.text(j, i, f'{v:.1f}%',
                  ha='center', va='center',
                  fontsize=FONT_SIZES['heatmap_cell'],
                  color=txt_col,
                  fontweight='bold' if is_diag else 'normal')

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

ax_c.text(-0.32, 1.04, 'C', transform=ax_c.transAxes,
          fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='bottom')

# ── Figure-level title ─────────────────────────────────────────

fig.suptitle('Supplementary Figure S4 — Error structure analysis (State × Modality × Valence)',
             fontsize=FONT_SIZES['subplot_title'], y=0.97)

# ════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════

os.makedirs(OUT_DIR, exist_ok=True)
stem = os.path.join(OUT_DIR, 'figS4_errorStructure')
fig.savefig(stem + '.pdf', dpi=300, bbox_inches='tight')
fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {stem}.pdf / .png')
