"""
Figure — Classification Performance and Error Structure
=======================================================
Two-row layout:

Panel a (top, full width):
    Model confusion matrices for the 2-class, 6-class, and 16-class tasks
    side by side (same Blues colormap / 0–100% normalisation as fig_accuracy_v7).
    Sub-labels i, ii, iii. Shared vertical colorbar on the right with chance-level
    dashed lines.

Panel b (bottom-left):
    F1 scores for all three tasks in a single grouped bar chart.
    One control bar + one model bar per task, with per-class dots in the
    shared class-colour scheme.

Panel c (bottom-centre):
    Error hierarchy for the 6-class task (State × Modality).
    3 bars: State, Modality, State × Modality — as % of total errors.
    50-run dots + mean line, same colour scheme as panel d.

Panel d (bottom-right):
    Error hierarchy for the 16-class task (State × Modality × Valence).
    7-bar breakdown identical to figS_classification_analysis panel a.

Shared legend (fig_accuracy_v7 style) at the bottom.

Usage (from repo root):
    python scripts/figures/run_figure_accuracy_error.py

Output:
    results/CombiPlots/{pdfs,pngs}/fig_accuracy_error.{pdf,png}
"""

import gc
import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logger import setup_logger
from src.visualization.figure_base import apply_style, FONT_SIZES, FIGURE_WIDTH, save_figure
from src.visualization.visualize_performance import (
    plot_confusion_matrix,
    _plot_class_symbol,
    draw_legend_panel,
    build_class_styles,
)
from src.utils.helpers import (
    load_all_results,
    get_style,
    load_h16_classification_reports,
)

apply_style()

logger = setup_logger(task_name='fig_accuracy_error',
                      log_dir='logs/run_figure_accuracy_error')

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_RESULTS_DIR = os.path.join('results', '_chkpt_finals')
OUT_DIR          = os.path.join('results', 'CombiPlots')

TASK_ORDER = [
    'MetabolicState_2',
    'State_Modality_6',
    'State_Modality_Valence_16',
]
TASK_LABELS = {
    'MetabolicState_2':          'State',
    'State_Modality_6':          'State ×\nModality',
    'State_Modality_Valence_16': 'State ×\nModality ×\nValence',
}
CM_SUB_LABELS = ['i', 'ii', 'iii']
CHANCE_LEVELS = [100 / 2, 100 / 6, 100 / 16]   # %, aligned with CM_SUB_LABELS

# ── 6-class error taxonomy ─────────────────────────────────────────────────────
# Index order matches TASK_CLASS_NAMES['State_Modality_6']:
#   "Odor (S)", "Odor (F)", "Taste (S)", "Taste (F)",
#   "Odor + Taste (S)", "Odor + Taste (F)"
CLASS_PROPS_6 = [
    ('Odor',     'Starved'),   # 0
    ('Odor',     'Fed'),       # 1
    ('Taste',    'Starved'),   # 2
    ('Taste',    'Fed'),       # 3
    ('Combined', 'Starved'),   # 4
    ('Combined', 'Fed'),       # 5
]

ERR6_KEYS    = ['State', 'Modality', 'State\n× Modality']
ERR6_COLOURS = ['#0072B2', '#009E73', '#D55E00']

# ── 16-class error taxonomy (mirrors run_sfigure_error_structure.py) ───────────
CLASS_PROPS_16 = [
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

ERR16_TYPE_ORDER = [1, 2, 3, 4, 6, 5, 7]
ERR16_XLABELS = [
    'Valence', 'State', 'Modality',
    'Valence\n× State', 'Valence\n× Modality',
    'State\n× Modality', 'Valence × State\n× Modality',
]
ERR16_COLOURS = {
    1: '#E69F00',
    2: '#0072B2',
    3: '#009E73',
    4: '#CC79A7',
    5: '#D55E00',
    6: '#56B4E9',
    7: '#999999',
}

# ── Error classification functions ─────────────────────────────────────────────

def _classify_error_6(i, j):
    mod_i, state_i = CLASS_PROPS_6[i]
    mod_j, state_j = CLASS_PROPS_6[j]
    if mod_i == mod_j:    return 'State'
    if state_i == state_j: return 'Modality'
    return 'State\n× Modality'


def _error_type_pct_6(cm):
    counts = {k: 0 for k in ERR6_KEYS}
    total_pred = int(np.array(cm).sum())
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            n = int(cm[i, j])
            if n == 0:
                continue
            counts[_classify_error_6(i, j)] += n
    if total_pred == 0:
        return {k: 0.0 for k in ERR6_KEYS}
    return {k: 100.0 * v / total_pred for k, v in counts.items()}


def _net_valence_16(cls_idx):
    _, _, val = CLASS_PROPS_16[cls_idx]
    if isinstance(val, tuple):
        return val[0] if val[0] == val[1] else 0
    return val


def _classify_error_16(i, j):
    mod_i, state_i, _ = CLASS_PROPS_16[i]
    mod_j, state_j, _ = CLASS_PROPS_16[j]
    same_mod   = (mod_i   == mod_j)
    same_state = (state_i == state_j)
    nv_i, nv_j = _net_valence_16(i), _net_valence_16(j)
    same_val   = (nv_i != 0 and nv_j != 0 and nv_i == nv_j)

    if same_mod:
        if same_state:                       return 1   # Valence only
        if same_val:                         return 2   # State only
        return 4                                        # State × Valence
    else:
        if same_state and same_val:          return 3   # Modality only
        if not same_state and same_val:      return 5   # State × Modality
        if same_state and not same_val:      return 6   # Modality × Valence
        return 7                                        # All three


def _error_type_pct_16(cm):
    counts = {t: 0 for t in range(1, 8)}
    total_pred = int(np.array(cm).sum())
    for i in range(16):
        for j in range(16):
            if i == j:
                continue
            n = int(cm[i, j])
            if n == 0:
                continue
            counts[_classify_error_16(i, j)] += n
    if total_pred == 0:
        return {t: 0.0 for t in range(1, 8)}
    return {t: 100.0 * v / total_pred for t, v in counts.items()}


# ── Data loaders ───────────────────────────────────────────────────────────────

def _load_h16_reports_and_cms(entry, n_classes):
    """Single-pass loader: open each H16 pkl once and extract both
    classification_report_dict and confusion_matrix.

    Returns (reports, cms) — avoids two separate passes over the same files.
    Only call for tasks that actually need confusion matrices (6- and 16-class).
    For tasks that only need reports, use load_h16_classification_reports instead.
    """
    all_runs = entry.get('runs', {}).get('H16', [])
    reports, cms = [], []
    for i, run in enumerate(all_runs):
        path = run.get('path')
        if path is None:
            continue
        if i % 10 == 0:
            logger.info(f'  loading pkl {i+1}/{len(all_runs)} …')
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            rpt = data.get('classification_report_dict')
            cm  = data.get('confusion_matrix')
            if rpt is not None:
                reports.append(rpt)
            if cm is not None and np.array(cm).shape == (n_classes, n_classes):
                cms.append(np.array(cm))
            del data
            gc.collect()
        except Exception as e:
            logger.warning(f'  {path}: {e}')
    return reports, cms


# ── Panel draw functions ───────────────────────────────────────────────────────

def _draw_f1_panel(ax, results_dict, all_reports_dict, class_styles_dict, rng):
    """One control+model bar pair per task, per-class colored dots on each bar."""
    bar_w  = 0.55
    grp_step = bar_w * 2 + 0.20
    grp_cx = {task: i * grp_step for i, task in enumerate(TASK_ORDER)}

    for task in TASK_ORDER:
        entry       = results_dict[task]
        all_reports = all_reports_dict[task]
        class_names = entry['__class_names__']
        c_styles    = class_styles_dict[task]
        ctrl_rpt    = entry['control']['classification_report_dict']

        # 50-run mean F1 per class (model); single-run for control
        f1_ctrl_bar  = [ctrl_rpt[cn]['f1-score'] for cn in class_names]
        f1_model_mean = [
            float(np.mean([r[cn]['f1-score'] for r in all_reports if cn in r]))
            for cn in class_names
        ]
        mean_f1_ctrl       = float(np.mean(f1_ctrl_bar))
        mean_f1_model_mean = float(np.mean(f1_model_mean))

        cx      = grp_cx[task]
        x_ctrl  = cx - bar_w / 2
        x_model = cx + bar_w / 2

        # Bars at 50-run mean height
        ax.bar(x_ctrl,  mean_f1_ctrl,       width=bar_w * 0.88,
               edgecolor='#b7bec4', facecolor='none', linewidth=2, zorder=2)
        ax.bar(x_model, mean_f1_model_mean, width=bar_w * 0.88,
               edgecolor='#094c80', facecolor='none', linewidth=2, zorder=2)

        n_cls = len(class_names)
        jit_m = rng.uniform(-bar_w * 0.28, bar_w * 0.28, size=n_cls)
        jit_c = rng.uniform(-bar_w * 0.28, bar_w * 0.28, size=n_cls)

        # Per-class colored dots: 50-run mean on model bar, single-run on control bar
        for k, f1 in enumerate(f1_model_mean):
            _plot_class_symbol(ax, x_model + jit_m[k], f1,
                               c_styles[k], markersize=5, zorder=5)

        for k, f1 in enumerate(f1_ctrl_bar):
            _plot_class_symbol(ax, x_ctrl + jit_c[k], f1,
                               c_styles[k], markersize=5, zorder=4)

    ax.set_xticks([grp_cx[t] for t in TASK_ORDER])
    ax.set_xticklabels(
        [TASK_LABELS[t] for t in TASK_ORDER],
        fontsize=FONT_SIZES['tick'], rotation=25, ha='right',
    )
    ax.tick_params(axis='x', length=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel('F1 Score', fontsize=FONT_SIZES['label'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _draw_hierarchy_bars(ax, xs, means, run_pcts_cols, colors, xlabels, rng,
                         xtick_rotation=35, xtick_fontsize=None, bar_w=0.55):
    """Shared bar-chart core used by panels c and d. Bars show 50-run mean."""
    if xtick_fontsize is None:
        xtick_fontsize = FONT_SIZES['tick']

    ax.bar(xs, means, width=bar_w, facecolor='none',
           edgecolor='#094c80', linewidth=2, zorder=2)

    if run_pcts_cols is not None and len(run_pcts_cols) > 0:
        for ki, (x, col) in enumerate(zip(xs, colors)):
            ys     = run_pcts_cols[:, ki]
            jitter = rng.uniform(-0.15, 0.15, size=len(ys))
            ax.scatter(np.full(len(ys), x) + jitter, ys,
                       s=22, color='#094c80', alpha=0.4, linewidths=0, zorder=4)

    for x, m in zip(xs, means):
        label_y = m + max(means) * 0.04
        ax.text(x, label_y, f'{m:.1f}%', ha='center', va='bottom',
                fontsize=FONT_SIZES['tick'], fontweight='bold',
                color='black', zorder=6)

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=xtick_fontsize,
                       rotation=xtick_rotation, ha='right')
    ax.set_ylabel('% of all predictions', fontsize=FONT_SIZES['label'])
    ymax = max(means) * 1.45 if max(means) > 0 else 10
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=False))
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds(xs[0], xs[-1])
    ax.tick_params(axis='x', length=0)


def _draw_hierarchy_6(ax, run_pcts, run_mean, rng):
    xs   = np.arange(3)
    means = run_mean   # shape (3,)
    cols  = np.column_stack([run_pcts[:, i] for i in range(3)]) \
            if len(run_pcts) > 0 else None
    _draw_hierarchy_bars(ax, xs, means, cols, ERR6_COLOURS, ERR6_KEYS, rng)


def _draw_hierarchy_16(ax, run_pcts, run_mean, rng):
    xs    = np.arange(7)
    means = np.array([run_mean[t - 1] for t in ERR16_TYPE_ORDER])
    colors = [ERR16_COLOURS[t] for t in ERR16_TYPE_ORDER]
    cols   = np.column_stack([run_pcts[:, t - 1] for t in ERR16_TYPE_ORDER]) \
             if len(run_pcts) > 0 else None
    # bar_w scaled so bars match visual width of panel c:
    # 0.55 × (n_c × col_c) / (n_d × col_d) = 0.55 × (3×1.0) / (7×2.8) ≈ 0.46
    _draw_hierarchy_bars(ax, xs, means, cols, colors, ERR16_XLABELS, rng,
                         xtick_rotation=45, bar_w=0.46)


def _draw_extra_legend(fig, ax_rect):
    """Add Control bar, Model bar, and 50-run mean line to the shared legend area."""
    ax = fig.add_axes(ax_rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    fs  = FONT_SIZES['legend_panel']
    col = '0.3'

    y_header = 0.95
    y_ctrl   = 0.68
    y_model  = 0.42
    y_mean   = 0.16

    ax.text(0.50, y_header, 'Key', ha='center', va='center',
            fontsize=fs, fontweight='bold', color='0.2')

    ax.add_patch(mpatches.Rectangle(
        (0.05, y_ctrl - 0.07), 0.18, 0.14,
        fill=False, edgecolor='#b7bec4', linewidth=2, clip_on=False,
    ))
    ax.text(0.30, y_ctrl, 'Control', ha='left', va='center', fontsize=fs, color=col)

    ax.add_patch(mpatches.Rectangle(
        (0.05, y_model - 0.07), 0.18, 0.14,
        fill=False, edgecolor='#094c80', linewidth=2, clip_on=False,
    ))
    ax.text(0.30, y_model, 'Model (50-run mean)', ha='left', va='center', fontsize=fs, color=col)


# ── Load data ─────────────────────────────────────────────────────────────────

logger.info('Loading styles …')
styles, TASK_CLASS_NAMES, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES, TASK_BICOLOR_INFO = \
    get_style('styles')

logger.info(f'Loading results from {BASE_RESULTS_DIR} …')
results = load_all_results(
    BASE_RESULTS_DIR, TASK_CLASS_NAMES,
    fixed_trf_for_E=16, fixed_cnn_for_H=16, only_cnn_dim=16,
)

all_reports_dict  = {}
class_styles_dict = {}
h16_cms_dict      = {}

# Tasks that need confusion matrices (panels c, d). MetabolicState_2 only
# needs classification reports (panel b), so skip the extra CM load there.
NEEDS_CMS = {'State_Modality_6', 'State_Modality_Valence_16'}

for task in TASK_ORDER:
    entry = results[task]
    n_cls = len(TASK_CLASS_NAMES[task])

    if task in NEEDS_CMS:
        logger.info(f'Loading reports + CMs for {task} …')
        reports, cms = _load_h16_reports_and_cms(entry, n_cls)
        h16_cms_dict[task] = cms
        logger.info(f'  → {len(reports)} reports, {len(cms)} CMs')
    else:
        logger.info(f'Loading reports for {task} …')
        reports = load_h16_classification_reports(entry)
        h16_cms_dict[task] = []
        logger.info(f'  → {len(reports)} reports')

    all_reports_dict[task] = reports
    entry['__h16_reports__'] = reports

    cs = build_class_styles(
        TASK_COLORS[task], TASK_EDGECOLORS[task], TASK_SHAPES[task],
        TASK_BICOLOR_INFO.get(task, {}),
    )
    class_styles_dict[task] = cs
    entry['__class_styles__'] = cs

# ── Error hierarchy statistics ─────────────────────────────────────────────────

# 6-class
entry_6   = results['State_Modality_6']
cm6_best  = np.array(entry_6['best']['confusion_matrix'])
cms_6     = h16_cms_dict['State_Modality_6']
best_pct_6 = _error_type_pct_6(cm6_best)
run_pcts_6 = np.array([[_error_type_pct_6(cm)[k] for k in ERR6_KEYS]
                        for cm in cms_6])
run_mean_6 = run_pcts_6.mean(axis=0) if len(run_pcts_6) > 0 else np.zeros(3)
logger.info('Computing 6-class error hierarchy …')
logger.info('6-class (best model): ' + '  '.join(f'{k}={best_pct_6[k]:.1f}%' for k in ERR6_KEYS))

# 16-class
entry_16   = results['State_Modality_Valence_16']
cm16_best  = np.array(entry_16['best']['confusion_matrix'])
cms_16     = h16_cms_dict['State_Modality_Valence_16']
best_pct_16 = _error_type_pct_16(cm16_best)
run_pcts_16 = np.array([[_error_type_pct_16(cm)[t] for t in range(1, 8)]
                         for cm in cms_16])
run_mean_16 = run_pcts_16.mean(axis=0) if len(run_pcts_16) > 0 else np.zeros(7)
logger.info('Computing 16-class error hierarchy …')
logger.info('16-class (best model): ' + '  '.join(f'T{t}={best_pct_16[t]:.1f}%' for t in ERR16_TYPE_ORDER))

# ── Figure layout ─────────────────────────────────────────────────────────────

logger.info('Assembling figure …')
RNG = np.random.default_rng(42)

fig = plt.figure(figsize=(FIGURE_WIDTH, 15))

gs_outer = GridSpec(
    2, 1, figure=fig,
    height_ratios=[0.6, 1.0],
    left=0.06, right=0.97,
    top=0.94, bottom=0.24,
    hspace=0.28,
)

# ── Top row: panel a (F1) left + panel b (3 CMs + colorbar) right ─────────────
gs_top = GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_outer[0],
    width_ratios=[1.0, 2.8],
    wspace=0.13,
)

# Panel a: F1 scores
ax_a = fig.add_subplot(gs_top[0])
_draw_f1_panel(ax_a, results, all_reports_dict, class_styles_dict, RNG)

# Panel b: 3 confusion matrices + colorbar (nested)
gs_cm = GridSpecFromSubplotSpec(
    1, 4, subplot_spec=gs_top[1],
    width_ratios=[1.0, 1.0, 1.0, 0.12],
    wspace=0.20,
)
ax_cm2  = fig.add_subplot(gs_cm[0])
ax_cm6  = fig.add_subplot(gs_cm[1])
ax_cm16 = fig.add_subplot(gs_cm[2])
ax_cbar = fig.add_subplot(gs_cm[3])

_cm_axes     = [ax_cm2, ax_cm6, ax_cm16]
_cm_labeling = ['x_axis', 'x_axis', 'x_axis']

for ax_cm, task, sub_lbl, axis_lbl in zip(
    _cm_axes, TASK_ORDER, CM_SUB_LABELS, _cm_labeling
):
    entry = results[task]
    plot_confusion_matrix(
        cl_name=f'{task} model',
        cm=entry['best']['confusion_matrix'],
        class_names=TASK_CLASS_NAMES[task],
        output_path=None,
        dataID=task,
        ax=ax_cm,
        annot=False,
        cbar=False,
        axis_labeling=axis_lbl,
        use_class_symbols=True,
        styles=styles,
        class_styles=entry['__class_styles__'],
    )
    ax_cm.set_title(sub_lbl, fontsize=FONT_SIZES['subplot_title'],
                    fontweight='bold', pad=4)

# Shared vertical colorbar with chance-level markers
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=100))
sm.set_array([])
cbar = plt.colorbar(sm, cax=ax_cbar, orientation='vertical')
cbar.set_label('Prediction (%)', fontsize=FONT_SIZES['colorbar'],
               rotation=270, labelpad=14)
cbar.set_ticks([0, 25, 50, 75, 100])
cbar.ax.tick_params(labelsize=FONT_SIZES['colorbar'])

trans_y = cbar.ax.get_yaxis_transform()
for v, lab in zip(CHANCE_LEVELS, CM_SUB_LABELS):
    cbar.ax.axhline(v, color='black', linestyle='--', linewidth=1, zorder=5)
    cbar.ax.text(1.7, v, lab, transform=trans_y,
                 ha='left', va='center',
                 fontsize=FONT_SIZES['colorbar'], fontweight='bold')

# ── Bottom row: panels c and d only ───────────────────────────────────────────
gs_bot = GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_outer[1],
    width_ratios=[1.0, 2.8],
    wspace=0.13,
)
ax_c = fig.add_subplot(gs_bot[0])
ax_d = fig.add_subplot(gs_bot[1])

_draw_hierarchy_6(ax_c, run_pcts_6, run_mean_6, RNG)
_draw_hierarchy_16(ax_d, run_pcts_16, run_mean_16, RNG)

# ── Panel labels ───────────────────────────────────────────────────────────────
_label_kw = dict(fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='top')

ax_a.text(  -0.18, 1.04, 'a.', transform=ax_a.transAxes,   **_label_kw)
ax_cm2.text(-0.18, 1.04, 'b.', transform=ax_cm2.transAxes, **_label_kw)
ax_c.text(  -0.18, 1.04, 'c.', transform=ax_c.transAxes,   **_label_kw)

# Place 'd.' at the same figure-x as 'b.' by computing real axis positions.
fig.canvas.draw()
pos_cm2 = ax_cm2.get_position()
pos_d   = ax_d.get_position()
fig_x_b = pos_cm2.x0 - 0.18 * pos_cm2.width
fig_y_d = pos_d.y1  + 0.04 * pos_d.height
fig.text(fig_x_b, fig_y_d, 'd.', transform=fig.transFigure, **_label_kw)

# ── Shared legend (fig_accuracy_v7 style) + Control/Model/mean key ───────────
draw_legend_panel(fig, styles, line_y=0.17, ax_rect=[0.03, 0.02, 0.77, 0.13])
_draw_extra_legend(fig, ax_rect=[0.83, 0.02, 0.15, 0.13])

# ── Save ──────────────────────────────────────────────────────────────────────
logger.info('Saving figure …')
os.makedirs(OUT_DIR, exist_ok=True)
save_figure(fig, os.path.join(OUT_DIR, 'fig_accuracy_error.pdf'))
logger.info('Done.')
