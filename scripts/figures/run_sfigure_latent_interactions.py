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
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path

from src.visualization.visualize_performance import (
    plot_1d_marginal,
    plot_2d_projection,
    _compute_biological_axes,
    _add_axis_indicator,
    draw_legend_panel,
)
from src.utils.helpers import get_style
from src.visualization.figure_base import apply_style, FONT_SIZES, FIGURE_WIDTH, save_figure

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

_BEST_KEYS = {'transformer_latent_space', 'latent_labels'}


def _load_best_pkl(task: str) -> dict:
    """Load only the keys we need from the most recent best pkl for a task.

    Bypasses load_all_results (which walks all run pkls) since this figure
    only needs the best checkpoint for two tasks.
    """
    best_dir = Path(BASE_RESULTS_DIR) / task / 'best'
    pkls = sorted(best_dir.glob('*.pkl'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pkls:
        raise FileNotFoundError(f"No best pkl found in {best_dir}")
    with pkls[0].open('rb') as f:
        data = pickle.load(f)
    return {k: data[k] for k in _BEST_KEYS if k in data}


def _add_1d_endpoint_labels(ax: plt.Axes, proj: np.ndarray, axis_display: str) -> None:
    line_extent = 1.0           # ← fixed; proj is expected pre-normalised to ±1
    neg, pos = _AXIS_ENDPOINTS.get(axis_display, ('', ''))
    if neg:
        ax.text(-line_extent, 0, f'{neg}  ', ha='right', va='center',
                fontsize=FONT_SIZES['annotation'])
    if pos:
        ax.text( line_extent, 0, f'  {pos}', ha='left',  va='center',
                fontsize=FONT_SIZES['annotation'])
    ax.plot([-line_extent, line_extent], [0, 0],
            color='gray', lw=1.2, alpha=0.6, zorder=0)
    ax.set_xlim(-line_extent, line_extent)   # padding for label text
    ax.set_title(axis_display, ha='center', fontsize=FONT_SIZES['label'], pad=4)

def plot_sfigure_latent_interactions(
    best_ii, best_iii,
    class_ii, class_iii,
    styles, colors, edges,
    bicolor_info=None,
    out_path=OUT_PATH,
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
    fig.text(panel_x, title_y[0], 'a',
             fontsize=FONT_SIZES['panel_label'], fontweight='bold')
    fig.text(0.50, title_y[0], 'ii. State, Modality',
             ha='center', va='bottom', fontsize=FONT_SIZES['title'], weight='bold')

    inner_a = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], hspace=0.55)

    for i, (key, display) in enumerate(TASK_AXES[TASK_II]):
        bbox = inner_a[i, 0].get_position(fig)
        ax = fig.add_axes([0.01, bbox.y0, 0.98, bbox.height])
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
        _add_1d_endpoint_labels(ax, proj_ii[key], display)

    # ── Panel b — task iii, 3 × 1D ────────────────────────────────────────
    fig.text(panel_x, title_y[1], 'b',
             fontsize=FONT_SIZES['panel_label'], fontweight='bold')
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
        _add_1d_endpoint_labels(ax, proj_iii[key], display)

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
    draw_legend_panel(fig, styles, line_y=0.17)

    # ── save ───────────────────────────────────────────────────────────────
    save_figure(fig, out_path)


def main():
    styles, TASK_CLASS_NAMES, TASK_COLORS, TASK_EDGECOLORS, _, TASK_BICOLOR_INFO = \
        get_style(style="styles")

    best_ii  = _load_best_pkl(TASK_II)
    best_iii = _load_best_pkl(TASK_III)

    plot_sfigure_latent_interactions(
        best_ii=best_ii,
        best_iii=best_iii,
        class_ii=TASK_CLASS_NAMES[TASK_II],
        class_iii=TASK_CLASS_NAMES[TASK_III],
        styles=styles,
        colors=TASK_COLORS,
        edges=TASK_EDGECOLORS,
        bicolor_info=TASK_BICOLOR_INFO,
        out_path=OUT_PATH,
    )


if __name__ == '__main__':
    main()
