"""
Figure — Classification Performance and Error Structure
=======================================================
Two-row layout:

Panel a (top, full width):
    Model confusion matrices for the 2-class, 6-class, and 16-class tasks
    side by side (same Blues colormap / 0–100% normalisation throughout).
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
    7-bar breakdown, 7-type taxonomy (see src/analysis/error_taxonomy.py).

Shared legend at the bottom.

Usage (from repo root):
    python -m scripts.figures.run_figure_accuracy_error

Output:
    results/CombiPlots/{pdfs,pngs}/fig_accuracy_error.{pdf,png}
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from src.utils.logger import setup_logger
from src.utils.config_loader import load_config
from src.visualization.figure_base import apply_style, FONT_SIZES, FIGURE_WIDTH, save_figure, add_panel_label
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
    load_h16_reports_and_cms,
)
from src.analysis.error_taxonomy import (
    ERR6_KEYS, ERR6_COLOURS,
    ERR16_TYPE_ORDER, ERR16_XLABELS, ERR16_COLOURS,
    MODALITY_IDX_16, T7_MODALITY_PAIRS,
    error_type_pct_6, error_type_pct_16,
    modality_error_breakdown_16, t7_modality_involvement_16,
    error_type_pct_16_subset,
)

apply_style()

# This figure's own x/y axis tick labels and axis titles run a bit larger
# than the shared FONT_SIZES defaults -- kept local rather than bumping the
# shared constants, which would affect every other figure too.
_TICK_FS  = FONT_SIZES['tick'] + 3
_LABEL_FS = FONT_SIZES['label'] + 3

# ── Configuration ──────────────────────────────────────────────────────────────
TASK_ORDER = [
    'MetabolicState_2',
    'State_Modality_6',
    'State_Modality_Valence_16',
]
TASK_LABELS = {
    'MetabolicState_2':          'State',
    'State_Modality_6':          'State\n× Modality',
    'State_Modality_Valence_16': 'State\n× Modality\n× Valence',
}
CM_SUB_LABELS = ['i', 'ii', 'iii']
CHANCE_LEVELS = [100 / 2, 100 / 6, 100 / 16]   # %, aligned with CM_SUB_LABELS

# Tasks that need confusion matrices (panels c, d). MetabolicState_2 only
# needs classification reports (panel b), so skip the extra CM load there.
NEEDS_CMS = {'State_Modality_6', 'State_Modality_Valence_16'}


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
        fontsize=_TICK_FS, rotation=25, ha='right',
    )
    ax.tick_params(axis='x', length=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel('F1 Score', fontsize=_LABEL_FS)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _draw_hierarchy_bars(ax, xs, means, run_pcts_cols, colors, xlabels, rng,
                         xtick_rotation=35, xtick_fontsize=None, bar_w=0.55):
    """Shared bar-chart core used by panels c and d. Bars show 50-run mean."""
    if xtick_fontsize is None:
        xtick_fontsize = _TICK_FS

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
                fontsize=_TICK_FS, fontweight='bold',
                color='black', zorder=6)

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=xtick_fontsize,
                       rotation=xtick_rotation, ha='right')
    ax.set_ylabel('% of all predictions', fontsize=_LABEL_FS)
    ymax = max(means) * 1.45 if max(means) > 0 else 10
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=False))
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
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


# ── Data loading ─────────────────────────────────────────────────────────────

def load_task_results(base_results_dir, task_class_names, task_colors,
                       task_edgecolors, task_shapes, task_bicolor_info, logger):
    """Load per-task results, classification reports, confusion matrices, and class styles."""
    results = load_all_results(
        base_results_dir, task_class_names,
        fixed_trf_for_E=16, fixed_cnn_for_H=16, only_cnn_dim=16,
    )

    all_reports_dict  = {}
    class_styles_dict = {}
    h16_cms_dict      = {}

    for task in TASK_ORDER:
        entry = results[task]
        n_cls = len(task_class_names[task])

        if task in NEEDS_CMS:
            logger.info(f'Loading reports + CMs for {task} …')
            reports, cms = load_h16_reports_and_cms(entry, n_cls, logger=logger)
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
            task_colors[task], task_edgecolors[task], task_shapes[task],
            task_bicolor_info.get(task, {}),
        )
        class_styles_dict[task] = cs
        entry['__class_styles__'] = cs

    return results, all_reports_dict, class_styles_dict, h16_cms_dict


# ── Error hierarchy statistics ────────────────────────────────────────────────

def compute_error_statistics(results, h16_cms_dict, logger):
    """Compute best-model and 50-run error-hierarchy percentages for the 6- and 16-class tasks."""
    entry_6   = results['State_Modality_6']
    cm6_best  = np.array(entry_6['best']['confusion_matrix'])
    cms_6     = h16_cms_dict['State_Modality_6']
    best_pct_6 = error_type_pct_6(cm6_best)
    run_pcts_6 = np.array([[error_type_pct_6(cm)[k] for k in ERR6_KEYS]
                            for cm in cms_6])
    run_mean_6 = run_pcts_6.mean(axis=0) if len(run_pcts_6) > 0 else np.zeros(3)
    logger.info('Computing 6-class error hierarchy …')
    logger.info('6-class (best model): ' + '  '.join(f'{k}={best_pct_6[k]:.1f}%' for k in ERR6_KEYS))

    entry_16   = results['State_Modality_Valence_16']
    cm16_best  = np.array(entry_16['best']['confusion_matrix'])
    cms_16     = h16_cms_dict['State_Modality_Valence_16']
    best_pct_16 = error_type_pct_16(cm16_best)
    run_pcts_16 = np.array([[error_type_pct_16(cm)[t] for t in range(1, 8)]
                             for cm in cms_16])
    run_mean_16 = run_pcts_16.mean(axis=0) if len(run_pcts_16) > 0 else np.zeros(7)
    logger.info('Computing 16-class error hierarchy …')
    logger.info('16-class (best model): ' + '  '.join(f'T{t}={best_pct_16[t]:.1f}%' for t in ERR16_TYPE_ORDER))

    return dict(
        best_pct_6=best_pct_6, run_pcts_6=run_pcts_6, run_mean_6=run_mean_6, cms_6=cms_6,
        best_pct_16=best_pct_16, run_pcts_16=run_pcts_16, run_mean_16=run_mean_16, cms_16=cms_16,
    )


def log_manuscript_diagnostics(task_class_names, all_reports_dict, stats, logger):
    """Log extra 16-class diagnostics used to check specific manuscript claims (log-only, not plotted).

    All computed from the same 50 loaded reports/CMs used for the plotted bars
    (mean across the 50 runs), not best-model-only.
    """
    class_names_16 = task_class_names['State_Modality_Valence_16']
    reports_16     = all_reports_dict['State_Modality_Valence_16']
    cms_16         = stats['cms_16']
    run_mean_16    = stats['run_mean_16']

    # 1. Per-class F1 (50-run mean), then Odor vs Taste class-group means.
    logger.info('─' * 60)
    logger.info('[1] 16-class per-class F1 (50-run mean):')
    f1_per_class_16 = {
        cn: float(np.mean([r[cn]['f1-score'] for r in reports_16 if cn in r]))
        for cn in class_names_16
    }
    for i, cn in enumerate(class_names_16):
        logger.info(f'  [{i:2d}] {cn}: F1={f1_per_class_16[cn]:.3f}')

    mean_f1_odor  = float(np.mean([f1_per_class_16[class_names_16[i]] for i in MODALITY_IDX_16['Odor']]))
    mean_f1_taste = float(np.mean([f1_per_class_16[class_names_16[i]] for i in MODALITY_IDX_16['Taste']]))
    logger.info(f'  Odor classes (n=4)  mean F1 = {mean_f1_odor:.3f}')
    logger.info(f'  Taste classes (n=4) mean F1 = {mean_f1_taste:.3f}')

    # 2. True-label = Taste / Odor: same-modality-diff-state vs different-modality-entirely.
    logger.info('─' * 60)
    logger.info('[2] Misclassification breakdown by true-label modality (50-run mean of per-run fractions):')
    for mod in ('Taste', 'Odor'):
        runs = [r for r in (modality_error_breakdown_16(cm, MODALITY_IDX_16[mod]) for cm in cms_16) if r is not None]
        if not runs:
            logger.info(f'  True-label = {mod}: no misclassifications found in any run.')
            continue
        mean_bd = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        logger.info(f'  True-label = {mod} (n={len(runs)} runs with >0 misclassifications):')
        logger.info(f"    same-modality/diff-state (wrong fed/starved) = {100*mean_bd['same_modality_diff_state']:.1f}%")
        logger.info(f"    same-modality/same-state (valence only)      = {100*mean_bd['same_modality_same_state_valence_only']:.1f}%")
        logger.info(f"    different-modality entirely                  = {100*mean_bd['different_modality']:.1f}%")

    # 3. T7 (Valence×State×Modality) errors: true-or-predicted modality involvement.
    logger.info('─' * 60)
    logger.info(f'[3] T7 (Valence×State×Modality, {run_mean_16[6]:.1f}%) — modality involvement '
                '(50-run mean of per-run fractions; true-or-pred label in category):')
    t7_runs = [r for r in (t7_modality_involvement_16(cm) for cm in cms_16) if r is not None]
    if t7_runs:
        logger.info(f'  involves Odor     = {100*float(np.mean([r["frac_odor"] for r in t7_runs])):.1f}%')
        logger.info(f'  involves Taste    = {100*float(np.mean([r["frac_taste"] for r in t7_runs])):.1f}%')
        logger.info(f'  involves Combined = {100*float(np.mean([r["frac_combined"] for r in t7_runs])):.1f}%')
        logger.info('  pairwise modality split:')
        for p in T7_MODALITY_PAIRS:
            mean_pair = float(np.mean([r['pair_fracs'][p] for r in t7_runs]))
            logger.info(f'    {p[0]} ↔ {p[1]} = {100*mean_pair:.1f}%')
    else:
        logger.info('  No T7 errors found in any run.')

    # 4. True-label = Combined only: 7-category decomposition restricted to that subset.
    logger.info('─' * 60)
    logger.info('[4] True-label = Combined only: 7-category error decomposition '
                '(50-run mean, % of Combined-true errors):')
    combined_runs = [r for r in (error_type_pct_16_subset(cm, MODALITY_IDX_16['Combined']) for cm in cms_16) if r is not None]
    if combined_runs:
        combined_mean = {t: float(np.mean([r[t] for r in combined_runs])) for t in range(1, 8)}
        for t in ERR16_TYPE_ORDER:
            label = ERR16_XLABELS[ERR16_TYPE_ORDER.index(t)].replace(chr(10), ' ')
            logger.info(f'  T{t} ({label}) = {combined_mean[t]:.1f}%')
    else:
        logger.info('  No Combined-true errors found in any run.')
    logger.info('─' * 60)


# ── Figure layout ─────────────────────────────────────────────────────────────

def build_figure(results, class_styles_dict, task_class_names, styles,
                  run_pcts_6, run_mean_6, run_pcts_16, run_mean_16,
                  all_reports_dict):
    """Assemble the full 2-row, 4-panel figure. Returns the Figure (not yet saved)."""
    rng = np.random.default_rng(42)

    fig = plt.figure(figsize=(FIGURE_WIDTH, 15))

    gs_outer = GridSpec(
        2, 1, figure=fig,
        height_ratios=[0.6, 1.0],
        left=0.06, right=0.97,
        top=0.94, bottom=0.28,
        hspace=0.24,
    )

    # ── Top row: panel a (F1) left + panel b (3 CMs + colorbar) right ─────────
    gs_top = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_outer[0],
        width_ratios=[1.0, 2.8],
        wspace=0.13,
    )

    # Panel a: F1 scores
    ax_a = fig.add_subplot(gs_top[0])
    _draw_f1_panel(ax_a, results, all_reports_dict, class_styles_dict, rng)

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
            class_names=task_class_names[task],
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
        # Bare numbering (i/ii/iii), positioned at the same axes-fraction
        # height as the colorbar's "Prediction (%)" title below
        # (set_label_coords y=1.10) so they align across the row.
        ax_cm.text(0.5, 1.10, sub_lbl, transform=ax_cm.transAxes,
                   ha='center', va='bottom',
                   fontsize=FONT_SIZES['title'], fontweight='bold')

    # Shared vertical colorbar with chance-level markers
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar, orientation='vertical')
    cbar.set_label('Prediction (%)', fontsize=FONT_SIZES['colorbar'],
                   rotation=0, labelpad=10)
    cbar.ax.yaxis.set_label_coords(0.5, 1.10)
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.ax.tick_params(labelsize=FONT_SIZES['colorbar'])

    trans_y = cbar.ax.get_yaxis_transform()
    for v, lab in zip(CHANCE_LEVELS, CM_SUB_LABELS):
        cbar.ax.axhline(v, color='black', linestyle='--', linewidth=1, zorder=5)
        cbar.ax.text(-0.7, v, lab, transform=trans_y,
                     ha='right', va='center',
                     fontsize=FONT_SIZES['colorbar'], fontweight='bold')

    # ── Bottom row: panels c and d only ───────────────────────────────────────
    gs_bot = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_outer[1],
        width_ratios=[1.0, 2.8],
        wspace=0.13,
    )
    ax_c = fig.add_subplot(gs_bot[0])
    ax_d = fig.add_subplot(gs_bot[1])

    _draw_hierarchy_6(ax_c, run_pcts_6, run_mean_6, rng)
    _draw_hierarchy_16(ax_d, run_pcts_16, run_mean_16, rng)

    # Top row sits above bottom row when rows overlap
    for _ax in [ax_a, ax_cm2, ax_cm6, ax_cm16, ax_cbar]:
        _ax.set_zorder(3)
        _ax.patch.set_visible(True)
    for _ax in [ax_c, ax_d]:
        _ax.set_zorder(1)

    # ── Panel labels ───────────────────────────────────────────────────────────
    add_panel_label(fig, 'a', ax=ax_a,   dx=-0.18, dy=0.04)
    add_panel_label(fig, 'b', ax=ax_cm2, dx=-0.18, dy=0.04)
    add_panel_label(fig, 'c', ax=ax_c,   dx=-0.18, dy=-0.04)

    # Force layout so axis positions and tick locations are finalised.
    fig.canvas.draw()

    # Place 'd' at the same figure-x as 'b'
    pos_cm2 = ax_cm2.get_position()
    pos_d   = ax_d.get_position()
    fig_x_b = pos_cm2.x0 - 0.18 * pos_cm2.width
    fig_y_d = pos_d.y1  - 0.04 * pos_d.height
    add_panel_label(fig, 'd', x=fig_x_b, y=fig_y_d)

    # Clip y-axis spine to last visible tick (panels c and d).
    for _ax in [ax_c, ax_d]:
        ylo, yhi = _ax.get_ylim()
        ticks = sorted(t for t in _ax.get_yticks() if ylo <= t <= yhi)
        if ticks:
            _ax.spines['left'].set_bounds(0, ticks[-1])

    # ── Shared legend + Control/Model/mean key ───────
    draw_legend_panel(fig, styles, line_y=0.17, ax_rect=[0.03, 0.02, 0.77, 0.13])
    _draw_extra_legend(fig, ax_rect=[0.83, 0.02, 0.15, 0.13])

    return fig


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    logger = setup_logger(task_name='fig_accuracy_error',
                          log_dir='logs/run_figure_accuracy_error')
    logger.info('Starting main')

    config = load_config()
    paths = config['paths']
    base_results_dir = paths['checkpoints_dir']
    out_dir = paths['output_dir']

    logger.info('Loading styles …')
    styles, TASK_CLASS_NAMES, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES, TASK_BICOLOR_INFO = \
        get_style('styles')

    logger.info(f'Loading results from {base_results_dir} …')
    results, all_reports_dict, class_styles_dict, h16_cms_dict = load_task_results(
        base_results_dir, TASK_CLASS_NAMES, TASK_COLORS, TASK_EDGECOLORS,
        TASK_SHAPES, TASK_BICOLOR_INFO, logger,
    )

    stats = compute_error_statistics(results, h16_cms_dict, logger)
    log_manuscript_diagnostics(TASK_CLASS_NAMES, all_reports_dict, stats, logger)

    logger.info('Assembling figure …')
    fig = build_figure(
        results, class_styles_dict, TASK_CLASS_NAMES, styles,
        stats['run_pcts_6'], stats['run_mean_6'],
        stats['run_pcts_16'], stats['run_mean_16'],
        all_reports_dict,
    )

    logger.info('Saving figure …')
    os.makedirs(out_dir, exist_ok=True)
    save_figure(fig, os.path.join(out_dir, 'fig_accuracy_error.pdf'))
    logger.info('Done.')


if __name__ == '__main__':
    main()
