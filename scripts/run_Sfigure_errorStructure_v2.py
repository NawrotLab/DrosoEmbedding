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
  1  Valence only        same modality, same state,  different valence
  2  State only          same modality, different state, same valence
  3  Modality only       different modality, same state,  same valence
Two-factor
  4  State × Valence     same modality, both differ
  5  State × Modality    different modality + different state, same valence  [cross-hierarchy]
  6  Modality × Valence  different modality + different valence, same state
Three-factor
  7  All three           all differ                                          [cross-hierarchy]

This script generates FOUR output figures:
  figS_errorStructure_v2        Compound panel (A = 7 types, B = per-modality, C = heatmap)
  figS_error_hierarchy          Standalone Panel A: 7-type bars (mean) + per-run dots
  figS_modality_errors          Per-modality error rates: bars (mean) + per-run dots
  figS_perclass_metrics         F1/Prec/Rec by group: bars (mean) + per-run dots
  figS_integrated_performance   Mean F1 bars split by within/cross error proportion,
                                per-class symbol dots, vertical class legend

Usage (from repo root):
    python scripts/run_Sfigure_errorStructure_v2.py

Output:
    results/CombiPlots/figS_errorStructure_v2.{pdf,png}
    results/CombiPlots/figS_error_hierarchy.{pdf,png}
    results/CombiPlots/figS_modality_errors.{pdf,png}
    results/CombiPlots/figS_perclass_metrics.{pdf,png}
    results/CombiPlots/figS_integrated_performance.{pdf,png}
"""

import gc
import os
import pickle
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization.figure_base import apply_style, FONT_SIZES
from src.utils.helpers import load_all_results, load_h16_classification_reports
from src.visualization.visualize_performance import _plot_class_symbol

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

METRICS         = ['f1-score', 'precision', 'recall']
METRIC_DISPLAY  = {'f1-score': 'F1', 'precision': 'Prec', 'recall': 'Rec'}

# Style keys aligned with CLASS_NAMES index order (for integrated figure)
STYLE_KEYS_16 = [
    'starved_odor_positive',      'starved_odor_negative',
    'fed_odor_positive',          'fed_odor_negative',
    'starved_taste_positive',     'starved_taste_negative',
    'fed_taste_positive',         'fed_taste_negative',
    'starved_odor_pos_taste_pos', 'starved_odor_neg_taste_neg',
    'starved_odor_neg_taste_pos', 'starved_odor_pos_taste_neg',
    'fed_odor_pos_taste_pos',     'fed_odor_neg_taste_neg',
    'fed_odor_neg_taste_pos',     'fed_odor_pos_taste_neg',
]

# ════════════════════════════════════════════════
# ERROR CLASSIFICATION
# ════════════════════════════════════════════════

def net_valence(cls_idx):
    """Net valence: +1/-1 for congruent/single; 0 for conflict Combined classes."""
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
    nv_i, nv_j = net_valence(i), net_valence(j)
    same_val   = (nv_i != 0 and nv_j != 0 and nv_i == nv_j)

    if same_mod:
        if same_state:  return 1
        if same_val:    return 2
        return 4
    else:
        if same_state and same_val:      return 3
        if not same_state and same_val:  return 5
        if same_state and not same_val:  return 6
        return 7


def error_type_pct(cm):
    """Compute % of total errors for each of the 7 types."""
    counts = {t: 0 for t in range(1, 8)}
    total  = 0
    for i in range(16):
        for j in range(16):
            if i == j: continue
            n = int(cm[i, j])
            if n == 0: continue
            counts[classify_error_7(i, j)] += n
            total += n
    if total == 0:
        return {t: 0.0 for t in range(1, 8)}
    return {t: 100.0 * v / total for t, v in counts.items()}


# ════════════════════════════════════════════════
# SHARED VISUALIZATION HELPER
# ════════════════════════════════════════════════

def _panel_bar_runs(ax, xs, means, run_vals, bar_colors, xlabels, ylabel,
                    bar_w=0.55, ylim_scale=1.40, rng=None):
    """
    Draw bars (heights = means) with per-run jittered dots overlaid.

    run_vals : (n_runs, n_bars) array, or None to skip dots.
    Returns max bar height (useful for annotations).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    means   = np.asarray(means)
    max_h   = float(means.max()) if len(means) > 0 else 1.0
    jit_hw  = bar_w * 0.30

    bars = ax.bar(xs, means, width=bar_w * 0.88, color=bar_colors,
                  alpha=0.75, edgecolor='none', zorder=2)

    if run_vals is not None and len(run_vals) > 0:
        run_vals = np.asarray(run_vals)
        for ki, (x, col) in enumerate(zip(xs, bar_colors)):
            ys     = run_vals[:, ki]
            jitter = rng.uniform(-jit_hw, jit_hw, size=len(ys))
            ax.scatter(np.full(len(ys), x) + jitter, ys,
                       s=14, color=col, alpha=0.35, linewidths=0, zorder=4)

    for bar, h in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, h + max_h * 0.025,
                f'{h:.1f}%', ha='center', va='bottom',
                fontsize=FONT_SIZES['small'], fontweight='bold')

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=FONT_SIZES['small'])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES['label'])
    ax.set_ylim(0, max_h * ylim_scale)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', length=0)
    return max_h


def _draw_vertical_legend(ax, all_styles):
    """Vertical class-symbol legend panel (Fed / Stv columns, grouped by modality)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ms = 9
    fs = FONT_SIZES['small']
    x_label, x_fed, x_stv = 0.02, 0.72, 0.90

    legend_groups = [
        ('Odor',  MODALITY_COLOURS['Odor'], [
            ('O app',       'fed_odor_positive',      'starved_odor_positive'),
            ('O avr',       'fed_odor_negative',       'starved_odor_negative'),
        ]),
        ('Taste', MODALITY_COLOURS['Taste'], [
            ('T app',       'fed_taste_positive',      'starved_taste_positive'),
            ('T avr',       'fed_taste_negative',       'starved_taste_negative'),
        ]),
        ('O+T',   MODALITY_COLOURS['Combined'], [
            ('OT app',         'fed_odor_pos_taste_pos',  'starved_odor_pos_taste_pos'),
            ('OT avr',         'fed_odor_neg_taste_neg',  'starved_odor_neg_taste_neg'),
            (r'T$^+$O$^-$',    'fed_odor_neg_taste_pos',  'starved_odor_neg_taste_pos'),
            (r'T$^-$O$^+$',    'fed_odor_pos_taste_neg',  'starved_odor_pos_taste_neg'),
        ]),
    ]

    y = 0.97
    ax.text(x_fed, y, 'Fed', ha='center', va='center', fontsize=fs, fontweight='bold', color='0.3')
    ax.text(x_stv, y, 'Stv', ha='center', va='center', fontsize=fs, fontweight='bold', color='0.3')

    y = 0.90
    for grp_name, grp_col, items in legend_groups:
        ax.text(x_label, y, grp_name, ha='left', va='center',
                fontsize=fs, fontweight='bold', color=grp_col)
        y -= 0.07
        for label, fed_key, stv_key in items:
            ax.text(x_label + 0.04, y, label, ha='left', va='center', fontsize=fs, color='0.3')
            _plot_class_symbol(ax, x_fed, y, all_styles.get(fed_key, {}), markersize=ms, zorder=5)
            _plot_class_symbol(ax, x_stv, y, all_styles.get(stv_key, {}), markersize=ms, zorder=5)
            y -= 0.065
        y -= 0.01

    ax.text(0.5, max(0.04, y + 0.01), u'● Fed   ○ Stv',
            ha='center', va='center', fontsize=fs - 1, color='0.5')


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
all_cms  = []
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

# ── 50-run classification reports (for per-class metrics) ─────────────────
print('Loading H16 classification reports …')
all_reports = load_h16_classification_reports(entry)
print(f'Loaded {len(all_reports)} classification reports.')

# ── Class styles + best-model per-class report (for integrated figure) ─────
_styles_path = os.path.join('src', 'visualization', 'styles.yaml')
with open(_styles_path, 'r') as _f:
    all_styles  = yaml.safe_load(_f)['styles']
class_styles = [all_styles.get(k, {}) for k in STYLE_KEYS_16]
best_report  = entry['best']['classification_report_dict']

# ════════════════════════════════════════════════
# COMPUTE STATISTICS
# ════════════════════════════════════════════════

# ── Error-type percentages ─────────────────────────────────────────────────
best_pct = error_type_pct(cm_raw)

run_pcts = np.array([[error_type_pct(cm)[t] for t in range(1, 8)]
                     for cm in all_cms])                       # (n_runs, 7)
run_mean = run_pcts.mean(axis=0) if len(run_pcts) > 0 else np.zeros(7)

# ── Per-modality error rates (best model + per-run) ───────────────────────
modality_stats = {}
for mod in MODALITY_ORDER:
    idx          = np.array(MODALITY_GROUPS[mod])
    N_total      = int(cm_raw[idx, :].sum())
    N_correct    = int(cm_raw[np.ix_(idx, idx)].diagonal().sum())
    N_errors     = N_total - N_correct
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

run_mod_err = np.zeros((len(all_cms), 3))
for ri, cm in enumerate(all_cms):
    for mi, mod in enumerate(MODALITY_ORDER):
        idx      = np.array(MODALITY_GROUPS[mod])
        N_tot    = int(cm[idx, :].sum())
        N_cor    = int(cm[np.ix_(idx, idx)].diagonal().sum())
        run_mod_err[ri, mi] = 100.0 * (N_tot - N_cor) / N_tot if N_tot > 0 else 0.0

mod_err_mean = (run_mod_err.mean(axis=0) if len(all_cms) > 0
                else np.array([modality_stats[m]['error_rate'] for m in MODALITY_ORDER]))

# ── Per-group metric averages (per-run, %) ────────────────────────────────
_grp_metric_lists = {mod: {m: [] for m in METRICS} for mod in MODALITY_ORDER}
for report in all_reports:
    for mod in MODALITY_ORDER:
        for metric in METRICS:
            vals = [report[CLASS_NAMES[i]][metric] * 100
                    for i in MODALITY_GROUPS[mod] if CLASS_NAMES[i] in report]
            if vals:
                _grp_metric_lists[mod][metric].append(float(np.mean(vals)))

run_grp_arr = {
    mod: {m: np.array(_grp_metric_lists[mod][m]) for m in METRICS}
    for mod in MODALITY_ORDER
}
grp_metric_mean = {
    mod: {m: run_grp_arr[mod][m].mean() if len(run_grp_arr[mod][m]) > 0 else 0.0
          for m in METRICS}
    for mod in MODALITY_ORDER
}

# ── 3×3 modality confusion ────────────────────────────────────────────────
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
cross_hier_best = best_pct[5] + best_pct[7]
cross_hier_mean = run_mean[4] + run_mean[6]   # 0-indexed: type5→idx4, type7→idx6

print('\n' + '=' * 65)
print('ERROR STRUCTURE SUMMARY  (7-type taxonomy)')
print('=' * 65)
print(f'Total misclassifications (best run): {total_errors}')
print()
for t in range(1, 8):
    tag   = ' ← cross-hierarchy' if t in (5, 7) else ''
    label = ERROR_TYPE_LABELS[t].replace('\n', ' ')
    print(f'  Type {t}  {label:30s}  best={best_pct[t]:5.1f}%  mean={run_mean[t-1]:5.1f}%{tag}')
print(f'\n  Cross-hierarchy (Types 5+7):  best={cross_hier_best:.1f}%  mean={cross_hier_mean:.1f}%')
print()
print('Per-modality error rates:')
for mi, mod in enumerate(MODALITY_ORDER):
    s = modality_stats[mod]
    print(f'  {mod:10s}  best={s["error_rate"]:5.1f}%  mean={mod_err_mean[mi]:.1f}%'
          f'  (within {s["within_pct"]:.1f}% / cross {s["cross_pct"]:.1f}%)')
print('=' * 65)

# ════════════════════════════════════════════════
# FIGURE 0: compound (unchanged from v1)
# ════════════════════════════════════════════════

fig = plt.figure(figsize=(14, 5))
gs  = GridSpec(1, 3, figure=fig,
               left=0.06, right=0.97, top=0.88, bottom=0.25,
               wspace=0.40, width_ratios=[2.2, 1.0, 0.95])
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])
ax_c = fig.add_subplot(gs[2])

rng0 = np.random.default_rng(42)

# ── Panel A ────────────────────────────────────────────────────────────────
xs      = np.arange(1, 8)
heights = [best_pct[t] for t in range(1, 8)]
colours = [ERROR_TYPE_COLOURS[t] for t in range(1, 8)]
max_h   = max(heights)

bars_a = ax_a.bar(xs, heights, width=0.55, color=colours, alpha=0.75,
                  edgecolor='none', zorder=2)

if len(run_pcts) > 0:
    for ti, t in enumerate(range(1, 8)):
        ys     = run_pcts[:, ti]
        jitter = rng0.uniform(-0.18, 0.18, size=len(ys))
        ax_a.scatter(np.full(len(ys), t) + jitter, ys,
                     s=14, color=ERROR_TYPE_COLOURS[t], alpha=0.35, linewidths=0, zorder=4)
        ax_a.plot([t - 0.22, t + 0.22], [run_mean[ti], run_mean[ti]],
                  color='black', linewidth=1.5, zorder=5)

for bar, h in zip(bars_a, heights):
    ax_a.text(bar.get_x() + bar.get_width() / 2, h + max_h * 0.025,
              f'{h:.1f}%', ha='center', va='bottom',
              fontsize=FONT_SIZES['small'], fontweight='bold')

y_bracket = max_h * 1.12
ax_a.annotate('', xy=(5, y_bracket), xytext=(7, y_bracket),
              arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.2))
ax_a.text(6, y_bracket + max_h * 0.03,
          f'Cross-hierarchy\n(Types 5+7): {cross_hier_best:.1f}%',
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

if len(run_pcts) > 0:
    mean_line = plt.Line2D([0], [0], color='black', linewidth=1.5, label='50-run mean')
    ax_a.legend(handles=[mean_line], fontsize=FONT_SIZES['small'] - 1,
                frameon=False, loc='upper left')

ax_a.text(-0.08, 1.04, 'A', transform=ax_a.transAxes,
          fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='bottom')

# ── Panel B ────────────────────────────────────────────────────────────────
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
    ax_b.text(x_b[i], s['error_rate'] + top * 0.025, f"{s['error_rate']:.1f}%",
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

# ── Panel C ────────────────────────────────────────────────────────────────
ax_c.imshow(mod_cm, vmin=0, vmax=100, cmap='Blues', aspect='auto')
for i in range(3):
    for j in range(3):
        v       = mod_cm[i, j]
        txt_col = 'white' if v > 58 else '#222222'
        ax_c.text(j, i, f'{v:.1f}%', ha='center', va='center',
                  fontsize=FONT_SIZES['heatmap_cell'], color=txt_col,
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

os.makedirs(OUT_DIR, exist_ok=True)
stem = os.path.join(OUT_DIR, 'figS_errorStructure_v2')
fig.savefig(stem + '.pdf', dpi=300, bbox_inches='tight')
fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {stem}.pdf / .png')

# ════════════════════════════════════════════════
# FIGURE 1: figS_error_hierarchy
# bars = 50-run mean; dots = individual runs
# ════════════════════════════════════════════════

fig1, ax1 = plt.subplots(figsize=(10, 5))
fig1.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.25)

rng1    = np.random.default_rng(42)
means_h = run_mean                                        # shape (7,)
runs_h  = run_pcts if len(run_pcts) > 0 else None        # shape (n_runs, 7)
colors_h = [ERROR_TYPE_COLOURS[t] for t in range(1, 8)]
xlabels_h = [ERROR_TYPE_LABELS[t] for t in range(1, 8)]

max_h1 = _panel_bar_runs(
    ax1, np.arange(1, 8), means_h, runs_h, colors_h, xlabels_h,
    ylabel='% of total errors', bar_w=0.55, ylim_scale=1.50, rng=rng1,
)

# Cross-hierarchy bracket
y_brk = max_h1 * 1.12
ax1.annotate('', xy=(5, y_brk), xytext=(7, y_brk),
             arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.2))
ax1.text(6, y_brk + max_h1 * 0.03,
         f'Cross-hierarchy\n(Types 5+7): {cross_hier_mean:.1f}%',
         ha='center', va='bottom',
         fontsize=FONT_SIZES['small'] - 1, color='#555555', style='italic')

stem = os.path.join(OUT_DIR, 'figS_error_hierarchy')
fig1.savefig(stem + '.pdf', dpi=300, bbox_inches='tight')
fig1.savefig(stem + '.png', dpi=300, bbox_inches='tight')
plt.close(fig1)
print(f'Saved: {stem}.pdf / .png')

# ════════════════════════════════════════════════
# FIGURE 2: figS_modality_errors
# bars = 50-run mean; dots = individual runs
# ════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(6, 5))
fig2.subplots_adjust(left=0.16, right=0.97, top=0.92, bottom=0.12)

rng2     = np.random.default_rng(42)
colors_m = [MODALITY_COLOURS[m] for m in MODALITY_ORDER]
runs_m   = run_mod_err if len(all_cms) > 0 else None

_panel_bar_runs(
    ax2, np.arange(3), mod_err_mean, runs_m, colors_m, MODALITY_ORDER,
    ylabel='Error rate (%)', bar_w=0.52, ylim_scale=1.35, rng=rng2,
)

stem = os.path.join(OUT_DIR, 'figS_modality_errors')
fig2.savefig(stem + '.pdf', dpi=300, bbox_inches='tight')
fig2.savefig(stem + '.png', dpi=300, bbox_inches='tight')
plt.close(fig2)
print(f'Saved: {stem}.pdf / .png')

# ════════════════════════════════════════════════
# FIGURE 3: figS_perclass_metrics
# Grouped bars: 3 groups × 3 metrics
# bars = 50-run mean; dots = individual runs; no arrow
# ════════════════════════════════════════════════

BAR_W_PM   = 0.18
GAP_PM     = 0.30
n_met      = len(METRICS)
grp_span   = n_met * BAR_W_PM
grp_step   = grp_span + GAP_PM
grp_cx     = {g: i * grp_step for i, g in enumerate(MODALITY_ORDER)}
met_off    = {m: (j - (n_met - 1) / 2) * BAR_W_PM for j, m in enumerate(METRICS)}

fig3, ax3 = plt.subplots(figsize=(9, 5))
fig3.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.28)

rng3 = np.random.default_rng(42)
trans3 = blended_transform_factory(ax3.transData, ax3.transAxes)

for group_name in MODALITY_ORDER:
    for metric in METRICS:
        x_bar    = grp_cx[group_name] + met_off[metric]
        mean_val = grp_metric_mean[group_name][metric]
        col      = MODALITY_COLOURS[group_name]

        ax3.bar(x_bar, mean_val, width=BAR_W_PM * 0.88,
                color=col, alpha=0.75, edgecolor='none', zorder=2)

        run_vals = run_grp_arr[group_name][metric]
        if len(run_vals) > 0:
            jitter = rng3.uniform(-BAR_W_PM * 0.30, BAR_W_PM * 0.30, size=len(run_vals))
            ax3.scatter(np.full(len(run_vals), x_bar) + jitter, run_vals,
                        s=14, color=col, alpha=0.35, linewidths=0, zorder=4)

        ax3.text(x_bar, mean_val + 1.2, f'{mean_val:.0f}',
                 ha='center', va='bottom',
                 fontsize=FONT_SIZES['small'], fontweight='bold', color=col)

        # Vertical metric label below x-axis
        ax3.text(x_bar, -0.04, f'{METRIC_DISPLAY[metric]} = {mean_val:.0f}',
                 ha='center', va='top', rotation=90,
                 fontsize=FONT_SIZES['small'], color=col,
                 transform=trans3, clip_on=False)

# Group name labels below metric text
for group_name in MODALITY_ORDER:
    ax3.text(grp_cx[group_name], -0.22, group_name,
             ha='center', va='top',
             fontsize=FONT_SIZES['label'], color=MODALITY_COLOURS[group_name],
             fontweight='bold', transform=trans3, clip_on=False)

# Vertical separators between groups
for a_grp, b_grp in [('Odor', 'Taste'), ('Taste', 'Combined')]:
    sep = (grp_cx[a_grp] + grp_cx[b_grp]) / 2
    ax3.axvline(sep, color='#CCCCCC', linewidth=1, zorder=1)

x_margin = BAR_W_PM * 2
ax3.set_xlim(grp_cx['Odor'] - grp_span / 2 - x_margin,
             grp_cx['Combined'] + grp_span / 2 + x_margin)
ax3.set_ylim(50, 105)
ax3.set_yticks(np.arange(50, 101, 10))
ax3.set_xticks([grp_cx[g] for g in MODALITY_ORDER])
ax3.set_xticklabels([''] * 3)
ax3.tick_params(axis='x', length=0)
ax3.set_ylabel('Score (%)', fontsize=FONT_SIZES['label'])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

stem = os.path.join(OUT_DIR, 'figS_perclass_metrics')
fig3.savefig(stem + '.pdf', dpi=300, bbox_inches='tight')
fig3.savefig(stem + '.png', dpi=300, bbox_inches='tight')
plt.close(fig3)
print(f'Saved: {stem}.pdf / .png')

# ════════════════════════════════════════════════
# FIGURE 4: figS_integrated_performance
# Stacked bars: height = group mean F1 (50-run mean)
# Bottom section (darker)  = within-modality error proportion of F1
# Top section (lighter)    = cross-modality error proportion of F1
# Dots = per-class F1 from best model, drawn as styled class symbols
# Right panel = vertical class legend
# ════════════════════════════════════════════════

def _lighten(hex_color, amount=0.55):
    """Mix color with white by `amount` (0 = original, 1 = white)."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


fig4 = plt.figure(figsize=(9, 5))
gs4  = GridSpec(1, 2, figure=fig4,
                width_ratios=[3.5, 1],
                left=0.10, right=0.99,
                bottom=0.12, top=0.94,
                wspace=0.06)
ax4     = fig4.add_subplot(gs4[0])
ax4_leg = fig4.add_subplot(gs4[1])

rng4  = np.random.default_rng(42)
xs4   = np.arange(len(MODALITY_ORDER))
bar_w4 = 0.55

for xi, mod in enumerate(MODALITY_ORDER):
    col        = MODALITY_COLOURS[mod]
    col_light  = _lighten(col, 0.55)
    mean_f1    = grp_metric_mean[mod]['f1-score']          # already in %
    within_p   = modality_stats[mod]['within_pct'] / 100   # fraction
    cross_p    = modality_stats[mod]['cross_pct']  / 100

    within_h = mean_f1 * within_p
    cross_h  = mean_f1 * cross_p

    # Bottom section — within-modality (darker)
    ax4.bar(xi, within_h, width=bar_w4, color=col,       zorder=2, label='_nolegend_')
    # Top section — cross-modality (lighter)
    ax4.bar(xi, cross_h,  width=bar_w4, color=col_light, zorder=2,
            bottom=within_h, label='_nolegend_')
    # Outline at total height
    ax4.bar(xi, mean_f1,  width=bar_w4, facecolor='none',
            edgecolor=col, linewidth=1.5, zorder=3)

    # Total F1 label above bar
    ax4.text(xi, mean_f1 + 1.5, f'{mean_f1:.1f}%',
             ha='center', va='bottom',
             fontsize=FONT_SIZES['small'], fontweight='bold', color=col)

    # Per-class F1 dots (best-model, styled class symbols)
    indices = MODALITY_GROUPS[mod]
    jitter  = rng4.uniform(-bar_w4 * 0.30, bar_w4 * 0.30, size=len(indices))
    for k, i_global in enumerate(indices):
        cls_f1 = best_report.get(CLASS_NAMES[i_global], {}).get('f1-score', np.nan)
        if np.isnan(cls_f1):
            continue
        _plot_class_symbol(ax4, xi + jitter[k], cls_f1 * 100,
                           class_styles[i_global], markersize=7, zorder=5)

# Within / cross legend patches (bottom-left)
within_patch = mpatches.Patch(color='#888888',                  label='Within-modality errors')
cross_patch  = mpatches.Patch(color=_lighten('#888888', 0.55),  label='Cross-modality errors')
ax4.legend(handles=[within_patch, cross_patch],
           fontsize=FONT_SIZES['small'] - 1, frameon=False,
           loc='lower right')

ax4.set_xticks(xs4)
ax4.set_xticklabels(MODALITY_ORDER, fontsize=FONT_SIZES['tick'])
ax4.set_ylabel('F1 score (%)', fontsize=FONT_SIZES['label'])
ax4.set_ylim(0, 115)
ax4.set_yticks(np.arange(0, 101, 20))
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.tick_params(axis='x', length=0)

_draw_vertical_legend(ax4_leg, all_styles)

stem = os.path.join(OUT_DIR, 'figS_integrated_performance')
fig4.savefig(stem + '.pdf', dpi=300, bbox_inches='tight')
fig4.savefig(stem + '.png', dpi=300, bbox_inches='tight')
plt.close(fig4)
print(f'Saved: {stem}.pdf / .png')
