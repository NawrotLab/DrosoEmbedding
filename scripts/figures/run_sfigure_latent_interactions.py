"""
Supplementary Figure — Latent Space Interactions
=================================================
Layout
------
  a)  Task ii  (State, Modality)          — 2 × 1D marginal plots, stacked
  b)  Task iii (State, Modality, Valence) — 3 × 1D marginal plots, stacked
  c)  Task iii — 3 × 2D pairwise projections side-by-side (height = a + b)

  Legend at bottom.

Reuses plot_1d_marginal, plot_2d_projection, _compute_biological_axes,
_add_axis_indicator, draw_legend_panel from visualize_performance.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from src.visualization.visualize_performance import (
    plot_1d_marginal,
    plot_2d_projection,
    _compute_biological_axes,
    _add_axis_indicator,
    draw_legend_panel,
)
from src.utils.helpers import get_style, load_or_build_all_results
from src.utils.config_loader import load_config
from src.visualization.figure_base import apply_style, FONT_SIZES, FIGURE_WIDTH, save_figure, add_panel_label

apply_style()

TASK_II  = 'State_Modality_6'
TASK_III = 'State_Modality_Valence_16'

# (data key, display name) per task — order determines subplot order
TASK_AXES = {
    TASK_II:  [('state', 'State'), ('modality', 'Modality')],
    TASK_III: [('state', 'State'), ('modality', 'Modality'), ('valence', 'Valence')],
}

# negative end → positive end, consistent with _compute_biological_axes sign convention
_AXIS_ENDPOINTS = {
    'State':    ('Starved', 'Fed'),
    'Modality': ('Odor',    'Taste'),
    'Valence':  ('Avers.',  'App.'),
}

def _add_1d_endpoint_labels(ax: plt.Axes, proj: np.ndarray, axis_display: str,
                            xlim_scale: float = 1.5) -> None:
    line_extent = 1.0
    neg, pos = _AXIS_ENDPOINTS.get(axis_display, ('', ''))
    if neg:
        ax.text(0.03, 0.5, f'{neg}  ', ha='left', va='center',
                fontsize=FONT_SIZES['annotation'], transform=ax.transAxes)
    if pos:
        ax.text(0.99, 0.5, f'  {pos}', ha='right', va='center',
                fontsize=FONT_SIZES['annotation'], transform=ax.transAxes)
    ax.axhline(0, xmin=0.08, xmax=0.95, color='gray', lw=0.8, alpha=0.6, zorder=0)
    ax.set_xlim(-line_extent * xlim_scale, line_extent * xlim_scale)
    ax.set_title(axis_display, ha='center', fontsize=FONT_SIZES['label'], pad=4)

def plot_sfigure_latent_interactions(
    best_ii, best_iii,
    class_ii, class_iii,
    styles, colors, edges,
    bicolor_info=None,
    out_path='results/CombiPlots/figS_latent_interactions.pdf',
):
    # ── projections ────────────────────────────────────────────────────────
    X_ii,  labels_ii  = best_ii['transformer_latent_space'],  best_ii['latent_labels']
    X_iii, labels_iii = best_iii['transformer_latent_space'], best_iii['latent_labels']

    bi_ii  = bicolor_info.get(TASK_II,  {}) if bicolor_info else {}
    bi_iii = bicolor_info.get(TASK_III, {}) if bicolor_info else {}

    proj_ii,  _, _ = _compute_biological_axes(X_ii,  labels_ii,  TASK_II,  class_ii)
    proj_iii, _, _ = _compute_biological_axes(X_iii, labels_iii, TASK_III, class_iii)

    # ── figure & outer grid ────────────────────────────────────────────────
    # height_ratios [2, 3, 5]: a gets 2 units, b gets 3 units, c = a + b = 5 units
    fig = plt.figure(figsize=(FIGURE_WIDTH, 14))

    outer = GridSpec(
        3, 1, figure=fig,
        left=0.06, right=0.96,
        top=0.93, bottom=0.18,
        hspace=0.50,
        height_ratios=[2, 3, 5],
    )

    panel_x = 0.04
    # approximate top-edge y positions for panel labels (figure fraction)
    title_y = [0.955, 0.705, 0.44]

    # ── Panel a — task ii, 2 × 1D ─────────────────────────────────────────
    add_panel_label(fig, 'a', x=panel_x, y=title_y[0], va='bottom')
    fig.text(0.50, title_y[0], 'ii. State, Modality',
             ha='center', va='bottom', fontsize=FONT_SIZES['title'], weight='bold')

    inner_a = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], hspace=0.55)

    for i, (key, display) in enumerate(TASK_AXES[TASK_II]):
        bbox = inner_a[i, 0].get_position(fig)
        ax = fig.add_axes([0.008, bbox.y0, 0.995, bbox.height])
        plot_1d_marginal(
            ax=ax,
            proj=proj_ii[key],
            class_names=class_ii,
            colors=colors[TASK_II],
            edges=edges[TASK_II],
            bicolor_info=bi_ii,
            axis_name='',
            s=180,
        )
        ax.lines[0].remove()  # drop plot_1d_marginal's short line; axhline replaces it
        _add_1d_endpoint_labels(ax, proj_ii[key], display, xlim_scale=3.4)

    # ── Panel b — task iii, 3 × 1D ────────────────────────────────────────
    add_panel_label(fig, 'b', x=panel_x, y=title_y[1], va='bottom')
    fig.text(0.50, title_y[1], 'iii. State, Modality, Valence',
             ha='center', va='bottom', fontsize=FONT_SIZES['title'], weight='bold')

    inner_b = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1], hspace=0.55)

    for i, (key, display) in enumerate(TASK_AXES[TASK_III]):
        bbox = inner_b[i, 0].get_position(fig)
        ax = fig.add_axes([0.01, bbox.y0 - 0.05, 0.98, bbox.height])
        plot_1d_marginal(
            ax=ax,
            proj=proj_iii[key],
            class_names=class_iii,
            colors=colors[TASK_III],
            edges=edges[TASK_III],
            bicolor_info=bi_iii,
            axis_name='',
            s=180,
        )
        ax.lines[0].remove()  # drop plot_1d_marginal's short line; axhline replaces it
        _add_1d_endpoint_labels(ax, proj_iii[key], display, xlim_scale=2.2)

    # ── Panel c — task iii, 3 × 2D pairwise ───────────────────────────────
    add_panel_label(fig, 'c', x=panel_x, y=title_y[2], va='bottom')
    fig.text(0.50, title_y[2], 'iii. State, Modality, Valence — pairwise projections',
             ha='center', va='bottom', fontsize=FONT_SIZES['title'], weight='bold')

    inner_c = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.35)

    pairs = [
        (proj_iii['state'],    proj_iii['modality'], 'State',    'Modality'),
        (proj_iii['state'],    proj_iii['valence'],  'State',    'Valence'),
        (proj_iii['modality'], proj_iii['valence'],  'Modality', 'Valence'),
    ]

    for j, (px, py, xlbl, ylbl) in enumerate(pairs):
        ax = fig.add_subplot(inner_c[0, j])
        plot_2d_projection(
            ax=ax,
            proj_x=px, proj_y=py,
            class_names=class_iii,
            colors=colors[TASK_III],
            edges=edges[TASK_III],
            bicolor_info=bi_iii,
            xlabel=xlbl,
            ylabel=ylbl,
            s=180,
        )
        # L-shaped axis indicator (xlbl → horizontal arm, ylbl → vertical arm)
        _add_axis_indicator(ax, [xlbl, ylbl])

    # ── legend ─────────────────────────────────────────────────────────────
    draw_legend_panel(fig, styles, line_y=0.17)

    # ── save ───────────────────────────────────────────────────────────────
    save_figure(fig, out_path)


def main():
    config = load_config()
    paths = config['paths']

    styles, TASK_CLASS_NAMES, TASK_COLORS, TASK_EDGECOLORS, _, TASK_BICOLOR_INFO = \
        get_style(style="styles")

    # "best" per task is auto-selected by load_all_results() (highest val
    # accuracy among E16_H16 runs); shares the aggregate cache with Figs 2/3.
    recompute = os.environ.get('EVAL_CACHE_RECOMPUTE', '').lower() in ('true', '1', 't')
    results = load_or_build_all_results(
        paths['eval_cache_path'], paths['eval_base_dir'], TASK_CLASS_NAMES,
        fixed_trf_for_E=16, fixed_cnn_for_H=16, only_cnn_dim=16,
        recompute=recompute,
    )
    best_ii  = results[TASK_II]['best']
    best_iii = results[TASK_III]['best']

    plot_sfigure_latent_interactions(
        best_ii=best_ii,
        best_iii=best_iii,
        class_ii=TASK_CLASS_NAMES[TASK_II],
        class_iii=TASK_CLASS_NAMES[TASK_III],
        styles=styles,
        colors=TASK_COLORS,
        edges=TASK_EDGECOLORS,
        bicolor_info=TASK_BICOLOR_INFO,
        out_path=os.path.join(paths['output_dir'], 'figS_latent_interactions.pdf'),
    )


if __name__ == '__main__':
    main()
