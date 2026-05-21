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
    build_class_styles,
)
from src.utils.helpers import load_all_results, get_style
from src.visualization.figure_base import apply_style, FONT_SIZES

apply_style()

BASE_RESULTS_DIR = os.path.join('results', '_chkpt_finals')
OUT_PATH = 'results/CombiPlots/fig_Sfigure_latent_interactions.pdf'

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


def _add_1d_endpoint_labels(ax: plt.Axes, proj: np.ndarray, axis_display: str) -> None:
    """Add directional end-labels to a 1D marginal plot and expand xlim to fit them.

    Must be called after plot_1d_marginal so the axis is already clean.
    xlim value mirrors the line extent computed inside plot_1d_marginal.
    """
    line_extent = float(np.abs(proj).max()) * 1.4
    neg, pos = _AXIS_ENDPOINTS.get(axis_display, ('', ''))
    if neg:
        ax.text(-line_extent, 0, f'{neg}  ', ha='right', va='center',
                fontsize=FONT_SIZES['annotation'])
    if pos:
        ax.text( line_extent, 0, f'  {pos}', ha='left',  va='center',
                fontsize=FONT_SIZES['annotation'])
    # widen xlim so the text isn't clipped
    ax.set_xlim(-line_extent * 2.0, line_extent * 2.0)


def plot_sfigure_latent_interactions(
    results_dict, styles, colors, edges, shapes,
    bicolor_info=None,
    out_path=OUT_PATH,
):
    # ── data ──────────────────────────────────────────────────────────────
    run_ii  = results_dict[TASK_II]
    run_iii = results_dict[TASK_III]

    best_ii  = run_ii.get('best', {})
    best_iii = run_iii.get('best', {})

    X_ii,  labels_ii  = best_ii.get('transformer_latent_space'),  best_ii.get('latent_labels')
    X_iii, labels_iii = best_iii.get('transformer_latent_space'), best_iii.get('latent_labels')

    class_ii  = run_ii['__class_names__']
    class_iii = run_iii['__class_names__']

    bi_ii  = bicolor_info.get(TASK_II,  {}) if bicolor_info else {}
    bi_iii = bicolor_info.get(TASK_III, {}) if bicolor_info else {}

    proj_ii,  _, _ = _compute_biological_axes(X_ii,  labels_ii,  TASK_II,  class_ii)
    proj_iii, _, _ = _compute_biological_axes(X_iii, labels_iii, TASK_III, class_iii)

    # ── figure & outer grid ────────────────────────────────────────────────
    # height_ratios [2, 3, 5]: a gets 2 units, b gets 3 units, c = a + b = 5 units
    fig = plt.figure(figsize=(18, 22))

    outer = GridSpec(
        3, 1, figure=fig,
        left=0.06, right=0.96,
        top=0.93, bottom=0.12,
        hspace=0.50,
        height_ratios=[2, 3, 5],
    )

    panel_x = 0.04
    # approximate top-edge y positions for panel labels (figure fraction)
    title_y = [0.955, 0.705, 0.44]

    # ── Panel a — task ii, 2 × 1D ─────────────────────────────────────────
    fig.text(panel_x, title_y[0], 'a',
             fontsize=FONT_SIZES['panel_label'], fontweight='bold')
    fig.text(0.50, title_y[0], 'ii. State, Modality',
             ha='center', va='bottom', fontsize=FONT_SIZES['title'], weight='bold')

    inner_a = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], hspace=0.55)

    for i, (key, display) in enumerate(TASK_AXES[TASK_II]):
        ax = fig.add_subplot(inner_a[i, 0])
        plot_1d_marginal(
            ax=ax,
            proj=proj_ii[key],
            class_names=class_ii,
            colors=colors[TASK_II],
            edges=edges[TASK_II],
            bicolor_info=bi_ii,
            axis_name='',   # suppressed — L-indicator used below instead
            s=180,
        )
        _add_1d_endpoint_labels(ax, proj_ii[key], display)
        _add_axis_indicator(ax, [display])

    # ── Panel b — task iii, 3 × 1D ────────────────────────────────────────
    fig.text(panel_x, title_y[1], 'b',
             fontsize=FONT_SIZES['panel_label'], fontweight='bold')
    fig.text(0.50, title_y[1], 'iii. State, Modality, Valence',
             ha='center', va='bottom', fontsize=FONT_SIZES['title'], weight='bold')

    inner_b = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1], hspace=0.55)

    for i, (key, display) in enumerate(TASK_AXES[TASK_III]):
        ax = fig.add_subplot(inner_b[i, 0])
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
        _add_1d_endpoint_labels(ax, proj_iii[key], display)
        _add_axis_indicator(ax, [display])

    # ── Panel c — task iii, 3 × 2D pairwise ───────────────────────────────
    fig.text(panel_x, title_y[2], 'c',
             fontsize=FONT_SIZES['panel_label'], fontweight='bold')
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
    draw_legend_panel(fig, styles, line_y=0.09)

    # ── save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stem = os.path.splitext(out_path)[0]
    for ext in ('pdf', 'png'):
        fig.savefig(f"{stem}.{ext}", bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {stem}.pdf + .png")


def main():
    styles, TASK_CLASS_NAMES, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES, TASK_BICOLOR_INFO = \
        get_style(style="styles")

    results_dict = load_all_results(
        BASE_RESULTS_DIR, TASK_CLASS_NAMES,
        fixed_trf_for_E=16, fixed_cnn_for_H=16,
        only_cnn_dim=16,
    )

    for task, result in results_dict.items():
        result['__class_styles__'] = build_class_styles(
            TASK_COLORS[task], TASK_EDGECOLORS[task], TASK_SHAPES[task],
            TASK_BICOLOR_INFO.get(task, {})
        )

    plot_sfigure_latent_interactions(
        results_dict, styles, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES,
        bicolor_info=TASK_BICOLOR_INFO,
        out_path=OUT_PATH,
    )


if __name__ == '__main__':
    main()
