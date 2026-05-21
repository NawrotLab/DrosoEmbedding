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

This script generates ONE output figure:
  figS_classification_analysis  Panel a = 7-type error hierarchy (bars + mean line + dots)
                                Panel b = per-modality F1/Prec/Rec grouped bars, class-colored
                                          dots, within/cross split, horizontal legend

Prints a full stats log (error types, per-modality and per-class accuracy) to stdout
for direct use in the main text.

Usage (from repo root):
    python scripts/run_Sfigure_errorStructure_v2.py

Output:
    results/CombiPlots/figS_classification_analysis.{pdf,png}
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization.figure_base import apply_style, FONT_SIZES, save_figure
from src.utils.helpers import load_all_results, load_h16_classification_reports
from src.visualization.visualize_performance import _plot_class_symbol, draw_legend_panel

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

METRICS = ['f1-score', 'precision', 'recall']

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


def _lighten(hex_color, amount=0.55):
    """Mix a hex color with white by `amount` (0 = original, 1 = white)."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _draw_hierarchy_panel(ax, rng):
    """
    Draw the 7-type error hierarchy bar chart onto ax.
    Type display order: 1,2,3,4,6,5,7  (groups single→two→three factor logically).
    Bars = best model; black line = 50-run mean; dots = individual runs;
    % labels at bar bottoms (black, 50-run mean values).
    """
    TYPE_ORDER = [1, 2, 3, 4, 6, 5, 7]
    xlabels = [
        'Valence', 'State', 'Modality',
        'Valence\n× State', 'Valence\n× Modality',
        'State\n× Modality', 'Valence × State\n× Modality',
    ]
    xs      = np.arange(len(TYPE_ORDER))
    heights = np.array([best_pct[t] for t in TYPE_ORDER])    # bars = best model
    means   = np.array([run_mean[t - 1] for t in TYPE_ORDER])  # line = 50-run mean
    colors  = [ERROR_TYPE_COLOURS[t] for t in TYPE_ORDER]
    bar_w   = 0.55

    ax.bar(xs, heights, width=bar_w, color=colors, alpha=0.75, edgecolor='none', zorder=2)

    # Per-run dots
    if len(run_pcts) > 0:
        runs_ro = np.column_stack([run_pcts[:, t - 1] for t in TYPE_ORDER])
        for ki, (x, col) in enumerate(zip(xs, colors)):
            ys     = runs_ro[:, ki]
            jitter = rng.uniform(-0.15, 0.15, size=len(ys))
            ax.scatter(np.full(len(ys), x) + jitter, ys,
                       s=14, color=col, alpha=0.35, linewidths=0, zorder=4)

    # Black horizontal line at 50-run mean
    hw = bar_w * 0.88 / 2
    for x, m in zip(xs, means):
        ax.plot([x - hw, x + hw], [m, m], color='black', linewidth=1.5, zorder=5)

    # % labels at bar bottoms — 50-run mean, black text
    for x, m in zip(xs, means):
        label_y = 0.8 if m > 2.5 else m + 0.3
        ax.text(x, label_y, f'{m:.1f}%', ha='center', va='bottom',
                fontsize=FONT_SIZES['small'] - 1, fontweight='bold',
                color='black', zorder=6)

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=FONT_SIZES['small'], rotation=35, ha='right')
    ax.set_ylabel('% of total errors', fontsize=FONT_SIZES['label'])
    ax.set_ylim(0, 60)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlim(-0.5, len(TYPE_ORDER) - 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', length=0)


def _draw_integrated_panel(ax, rng):
    """
    Draw integrated performance panel: 9 grouped bars (3 modalities × 3 metrics).
    Bars: stacked within/cross sections, height = best-model value.
    Black line = 50-run mean. Dots = per-class 50-run means, colored by class style.
    Legend is drawn separately via draw_legend_panel + _draw_extra_legend_items.
    """
    METRIC_ORDER  = ['f1-score', 'precision', 'recall']
    METRIC_LABELS = {'f1-score': 'F1', 'precision': 'Prec', 'recall': 'Rec'}
    BAR_W    = 0.20
    GRP_GAP  = 0.32
    grp_span = len(METRIC_ORDER) * BAR_W
    grp_step = grp_span + GRP_GAP
    grp_cx   = {mod: i * grp_step for i, mod in enumerate(MODALITY_ORDER)}
    met_off  = {m: (j - (len(METRIC_ORDER) - 1) / 2) * BAR_W
                for j, m in enumerate(METRIC_ORDER)}

    cls_mean_by_metric = {
        'f1-score':  cls_f1_mean,
        'precision': cls_prec_mean,
        'recall':    cls_rec_mean,
    }
    best_grp_by_metric = {
        'f1-score':  best_grp_f1,
        'precision': best_grp_prec,
        'recall':    best_grp_rec,
    }

    for mod in MODALITY_ORDER:
        col    = MODALITY_COLOURS[mod]
        col_lt = _lighten(col, 0.55)
        w_frac = modality_stats[mod]['within_pct'] / 100
        c_frac = modality_stats[mod]['cross_pct']  / 100

        for metric in METRIC_ORDER:
            x_bar    = grp_cx[mod] + met_off[metric]
            best_val = best_grp_by_metric[metric][mod]
            mean_val = grp_metric_mean[mod][metric]
            hw       = BAR_W * 0.88 / 2

            ax.bar(x_bar, best_val * w_frac, width=BAR_W * 0.88,
                   color=col, zorder=2)
            ax.bar(x_bar, best_val * c_frac, width=BAR_W * 0.88,
                   color=col_lt, zorder=2, bottom=best_val * w_frac)
            ax.bar(x_bar, best_val, width=BAR_W * 0.88,
                   facecolor='none', edgecolor=col, linewidth=1.0, zorder=3)

            # Black 50-run mean line (matching Panel A linewidth)
            ax.plot([x_bar - hw, x_bar + hw], [mean_val, mean_val],
                    color='black', linewidth=1.5, zorder=5)

            # Per-class dots colored by class style (filled=Fed, open=Starved)
            valid_idx = [i for i in MODALITY_GROUPS[mod]
                         if not np.isnan(cls_mean_by_metric[metric][i])]
            if valid_idx:
                jitter = rng.uniform(-BAR_W * 0.28, BAR_W * 0.28, size=len(valid_idx))
                for k, cls_i in enumerate(valid_idx):
                    _plot_class_symbol(ax, x_bar + jitter[k],
                                       cls_mean_by_metric[metric][cls_i],
                                       class_styles[cls_i], markersize=5, zorder=6)

            # Metric label + mean: vertical, rising from x-axis up inside bar
            ax.text(x_bar, 50.5, f'{METRIC_LABELS[metric]}  {mean_val:.0f}%',
                    ha='center', va='bottom', rotation=90,
                    fontsize=FONT_SIZES['small'], fontweight='bold',
                    color='black', zorder=7)

    # Vertical group separators
    for a_grp, b_grp in [('Odor', 'Taste'), ('Taste', 'Combined')]:
        sep = (grp_cx[a_grp] + grp_cx[b_grp]) / 2
        ax.axvline(sep, color='#CCCCCC', linewidth=0.8, zorder=1)

    x_margin = BAR_W * 1.5
    ax.set_xlim(grp_cx['Odor'] - grp_span / 2 - x_margin,
                grp_cx['Combined'] + grp_span / 2 + x_margin)
    ax.set_ylim(50, 100)
    ax.set_yticks(np.arange(50, 101, 10))
    ax.set_xticks([grp_cx[mod] for mod in MODALITY_ORDER])
    ax.set_xticklabels(MODALITY_ORDER, fontsize=FONT_SIZES['small'], rotation=35, ha='right')
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('Accuracy [%]', fontsize=FONT_SIZES['label'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _draw_extra_legend_items(fig, ax_rect):
    """
    Draw the figure-specific legend items (mean line + within/cross swatches)
    into a small axes panel, vertically aligned with draw_legend_panel rows.
    """
    ax = fig.add_axes(ax_rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    fs  = FONT_SIZES['legend_panel']
    col = '0.3'

    # Align to the same y positions draw_legend_panel uses
    y_header = 0.95
    y_row1   = 0.65   # aligns with Fed row
    y_row2   = 0.42
    y_row3   = 0.18

    ax.text(0.50, y_header, 'Key', ha='center', va='center',
            fontsize=fs, fontweight='bold', color='0.2')

    # 50-run mean line
    ax.plot([0.05, 0.22], [y_row1, y_row1], color='black', linewidth=1.5,
            solid_capstyle='butt', clip_on=False)
    ax.text(0.26, y_row1, '50-run mean', ha='left', va='center', fontsize=fs, color=col)

    # Within-modality swatch
    ax.add_patch(mpatches.Rectangle((0.05, y_row2 - 0.06), 0.15, 0.12,
                                    facecolor='#888888', edgecolor='none', clip_on=False))
    ax.text(0.26, y_row2, 'Within-mod.', ha='left', va='center', fontsize=fs, color=col)

    # Cross-modality swatch (lightened)
    ax.add_patch(mpatches.Rectangle((0.05, y_row3 - 0.06), 0.15, 0.12,
                                    facecolor=_lighten('#888888', 0.55),
                                    edgecolor='none', clip_on=False))
    ax.text(0.26, y_row3, 'Cross-mod.', ha='left', va='center', fontsize=fs, color=col)


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

# ── Per-class 50-run mean F1 (for integrated figure dots) ─────────────────
cls_f1_mean = {}
for _i in range(16):
    _vals = [r[CLASS_NAMES[_i]]['f1-score'] * 100
             for r in all_reports if CLASS_NAMES[_i] in r]
    cls_f1_mean[_i] = float(np.mean(_vals)) if _vals else np.nan

# ── Best-model group-level F1 (for integrated figure bars) ────────────────
best_grp_f1 = {
    mod: float(np.mean([
        best_report[CLASS_NAMES[i]]['f1-score'] * 100
        for i in MODALITY_GROUPS[mod] if CLASS_NAMES[i] in best_report
    ]))
    for mod in MODALITY_ORDER
}

# ── Per-class 50-run mean Precision and Recall ─────────────────────────────
cls_prec_mean = {}
cls_rec_mean  = {}
for _i in range(16):
    _prec = [r[CLASS_NAMES[_i]]['precision'] * 100 for r in all_reports if CLASS_NAMES[_i] in r]
    _rec  = [r[CLASS_NAMES[_i]]['recall']    * 100 for r in all_reports if CLASS_NAMES[_i] in r]
    cls_prec_mean[_i] = float(np.mean(_prec)) if _prec else np.nan
    cls_rec_mean[_i]  = float(np.mean(_rec))  if _rec  else np.nan

# ── Best-model group-level Precision and Recall ────────────────────────────
best_grp_prec = {
    mod: float(np.mean([best_report[CLASS_NAMES[i]]['precision'] * 100
                        for i in MODALITY_GROUPS[mod] if CLASS_NAMES[i] in best_report]))
    for mod in MODALITY_ORDER
}
best_grp_rec = {
    mod: float(np.mean([best_report[CLASS_NAMES[i]]['recall'] * 100
                        for i in MODALITY_GROUPS[mod] if CLASS_NAMES[i] in best_report]))
    for mod in MODALITY_ORDER
}


# ════════════════════════════════════════════════
# PRINT SUMMARY
# ════════════════════════════════════════════════

total_errors    = sum(int(cm_raw[i, j]) for i in range(16) for j in range(16) if i != j)
cross_hier_best = best_pct[5] + best_pct[7]
cross_hier_mean = run_mean[4] + run_mean[6]   # type5→idx4, type7→idx6

W = 72  # print width

print('\n' + '═' * W)
print('  STATS LOG — figS_classification_analysis')
print('  (all values suitable for direct citation in main text)')
print('═' * W)

# ── Error structure ────────────────────────────────────────────────────────
print(f'\n{"─"*W}')
print('  ERROR STRUCTURE  (7-type taxonomy, % of total misclassifications)')
print(f'{"─"*W}')
print(f'  Runs loaded : {len(all_cms)}  |  Total misclassifications (best model): {total_errors}')
print()
TYPE_ORDER_DISPLAY = [1, 2, 3, 4, 6, 5, 7]
for t in TYPE_ORDER_DISPLAY:
    label = ERROR_TYPE_LABELS[t].replace('\n', ' ')
    tag   = '  ← cross-hierarchy' if t in (5, 7) else ''
    print(f'  Type {t}  {label:28s}  best={best_pct[t]:5.1f}%   50-run mean={run_mean[t-1]:5.1f}%{tag}')
print()
print(f'  Cross-hierarchy combined (Types 5+7):')
print(f'    best model = {cross_hier_best:.1f}%   50-run mean = {cross_hier_mean:.1f}%')

# ── Per-modality accuracy ──────────────────────────────────────────────────
print(f'\n{"─"*W}')
print('  PER-MODALITY ACCURACY  (group mean across classes, %)')
print(f'{"─"*W}')
print(f'  {"Modality":10s}  {"Metric":10s}  {"Best model":>12s}  {"50-run mean":>12s}  {"SD":>8s}')
for mod in MODALITY_ORDER:
    for metric in METRICS:
        best_val  = {'f1-score': best_grp_f1, 'precision': best_grp_prec,
                     'recall':   best_grp_rec}[metric][mod]
        mean_val  = grp_metric_mean[mod][metric]
        arr       = run_grp_arr[mod][metric]
        sd        = float(arr.std()) if len(arr) > 1 else float('nan')
        print(f'  {mod:10s}  {metric:10s}  {best_val:>12.1f}  {mean_val:>12.1f}  {sd:>8.1f}')
    print()

# ── Per-modality error breakdown ───────────────────────────────────────────
print(f'{"─"*W}')
print('  PER-MODALITY ERROR BREAKDOWN  (best model)')
print(f'{"─"*W}')
print(f'  {"Modality":10s}  {"Error rate":>12s}  {"Within-mod %":>14s}  {"Cross-mod %":>13s}')
for mod in MODALITY_ORDER:
    s = modality_stats[mod]
    print(f'  {mod:10s}  {s["error_rate"]:>12.1f}  {s["within_pct"]:>14.1f}  {s["cross_pct"]:>13.1f}')

# ── Per-class 50-run mean metrics ─────────────────────────────────────────
print(f'\n{"─"*W}')
print('  PER-CLASS 50-RUN MEAN METRICS  (%)')
print(f'{"─"*W}')
print(f'  {"Class":28s}  {"Modality":10s}  {"F1":>8s}  {"Precision":>10s}  {"Recall":>8s}')
for _i in range(16):
    mod = CLASS_PROPS[_i][0]
    print(f'  {CLASS_NAMES[_i]:28s}  {mod:10s}'
          f'  {cls_f1_mean[_i]:>8.1f}  {cls_prec_mean[_i]:>10.1f}  {cls_rec_mean[_i]:>8.1f}')

print('\n' + '═' * W + '\n')

# ════════════════════════════════════════════════
# FIGURE 5: figS_classification_analysis
# Panel A = error hierarchy  |  Panel B = integrated performance + legend
# ════════════════════════════════════════════════

fig5 = plt.figure(figsize=(14, 7))
gs5  = GridSpec(1, 2, figure=fig5,
                width_ratios=[2.6, 1],
                left=0.06, right=0.99,
                bottom=0.43, top=0.94,
                wspace=0.38)
ax5a = fig5.add_subplot(gs5[0])
ax5b = fig5.add_subplot(gs5[1])

_draw_hierarchy_panel(ax5a, np.random.default_rng(42))
_draw_integrated_panel(ax5b, np.random.default_rng(42))
draw_legend_panel(fig5, all_styles, line_y=0.3, ax_rect=[0.03, 0.02, 0.78, 0.25])
_draw_extra_legend_items(fig5, ax_rect=[0.82, 0.02, 0.16, 0.25])

ax5a.text(-0.08, 1.04, 'a', transform=ax5a.transAxes,
          fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='bottom')
ax5b.text(-0.12, 1.04, 'b', transform=ax5b.transAxes,
          fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='bottom')

stem = os.path.join(OUT_DIR, 'figS_classification_analysis')
save_figure(fig5, stem + '.pdf')
