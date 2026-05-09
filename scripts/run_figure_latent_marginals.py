"""
Figure: Marginal 1D Latent Projections
=======================================
One task group per section; subplots stacked vertically (full figure width each).

  Group i   (State)                   : title + 1 plot
  Group ii  (State, Modality)         : title + 2 plots stacked
  Group iii (State, Modality, Valence): title + 3 plots stacked

Reuses plot_1d_marginal() and _compute_biological_axes() from
visualize_performance.py. Marker/colour conventions match figure_latent.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

from src.visualization.visualize_performance import (
    plot_1d_marginal,
    plot_2d_projection,
    _compute_biological_axes,
    draw_legend_panel,
    build_class_styles,
)
from src.utils.helpers import load_all_results, get_style
from src.visualization.figure_base import apply_style, FONT_SIZES

apply_style()

BASE_RESULTS_DIR = os.path.join('results', '_chkpt_finals')
OUT_PATH = 'results/CombiPlots/fig_latent_marginals.pdf'

TASKS = [
    'MetabolicState_2',
    'State_Modality_6',
    'State_Modality_Valence_16',
]
TASK_NAMES = ['i. State', 'ii. State, Modality', 'iii. State, Modality, Valence']
TASK_AXES = {
    'MetabolicState_2':          ['state'],
    'State_Modality_6':          ['state', 'modality'],
    'State_Modality_Valence_16': ['state', 'modality', 'valence'],
}
AXIS_DISPLAY = {'state': 'State', 'modality': 'Modality', 'valence': 'Valence'}


def plot_figure_marginals(
    results_dict, styles, colors, edges, shapes,
    bicolor_info=None,
    out_path=OUT_PATH,
):
    fig = plt.figure(figsize=(18, 18))

    # Outer grid: 3 task groups, height proportional to subplot count (1 : 2 : 3).
    # Extra hspace between groups; inner hspace keeps plots within a group tight.
    outer = GridSpec(3, 1, figure=fig,
                     left=0.03, right=0.99,
                     top=0.93, bottom=0.20,
                     hspace=0.55,
                     height_ratios=[1, 2, 3])

    # y positions for task group titles (placed above each outer row)
    title_y = [0.955, 0.72, 0.44]

    fig.text(0.53, 0.975, 'Centroid projections — marginal 1D axes',
             ha='center', va='center',
             fontsize=FONT_SIZES['title'], weight='bold')

    for row, (task, task_name) in enumerate(zip(TASKS, TASK_NAMES)):
        run_dict = results_dict.get(task)
        if run_dict is None:
            continue

        class_names = run_dict['__class_names__']
        task_bicolor = bicolor_info.get(task, {}) if bicolor_info else {}

        best_data = run_dict.get('best', {})
        X = best_data.get('transformer_latent_space')
        labels = best_data.get('latent_labels')
        if X is None or labels is None:
            continue

        projections, _, _ = _compute_biological_axes(X, labels, task, class_names)
        axis_keys = TASK_AXES[task]
        n_ax = len(axis_keys)

        # Task group title above the group
        fig.text(0.53, title_y[row], task_name,
                 ha='center', va='bottom', fontsize=FONT_SIZES['title'],
                 weight='bold', transform=fig.transFigure)

        # Stack subplots vertically within this task group
        inner = GridSpecFromSubplotSpec(n_ax, 1, subplot_spec=outer[row], hspace=0.45)

        for col_idx, ax_key in enumerate(axis_keys):
            ax = fig.add_subplot(inner[col_idx, 0])
            plot_1d_marginal(
                ax=ax,
                proj=projections[ax_key],
                class_names=class_names,
                colors=colors[task],
                edges=edges[task],
                bicolor_info=task_bicolor,
                axis_name=AXIS_DISPLAY[ax_key],
                s=180,
            )

    draw_legend_panel(fig, styles, line_y=0.165)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_figure_pairwise_2d(
    results_dict, styles, colors, edges,
    bicolor_info=None,
    out_path='results/CombiPlots/fig_latent_pairwise_2d.pdf',
):
    """3 side-by-side 2D scatter plots for task iii: all pairwise axis combinations."""
    task = 'State_Modality_Valence_16'
    run_dict = results_dict.get(task)
    if run_dict is None:
        print("Task iii data not found — skipping pairwise 2D figure.")
        return

    class_names = run_dict['__class_names__']
    task_bicolor = bicolor_info.get(task, {}) if bicolor_info else {}
    best_data = run_dict.get('best', {})
    X = best_data.get('transformer_latent_space')
    labels = best_data.get('latent_labels')
    if X is None or labels is None:
        print("No latent data for task iii — skipping pairwise 2D figure.")
        return

    projections, _, _ = _compute_biological_axes(X, labels, task, class_names)
    proj_s = projections['state']
    proj_m = projections['modality']
    proj_v = projections['valence']

    pairs = [
        (proj_s, proj_m, 'State',    'Modality'),
        (proj_s, proj_v, 'State',    'Valence'),
        (proj_m, proj_v, 'Modality', 'Valence'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    fig.text(0.53, 0.97, 'iii. State, Modality, Valence — pairwise 2D projections',
             ha='center', va='top', fontsize=FONT_SIZES['title'], weight='bold')

    for ax, (px, py, xlbl, ylbl) in zip(axes, pairs):
        plot_2d_projection(
            ax=ax,
            proj_x=px, proj_y=py,
            class_names=class_names,
            colors=colors[task],
            edges=edges[task],
            bicolor_info=task_bicolor,
            xlabel=xlbl,
            ylabel=ylbl,
            s=180,
        )

    plt.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.18, wspace=0.3)
    draw_legend_panel(fig, styles, line_y=0.145)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_3d_rotating(
    projections, class_names, colors, edges,
    out_path='results/CombiPlots/fig_latent_marginals_3d.gif',
    fps=20, n_frames=180,
):
    """Save a rotating 3D scatter of task-iii centroids (State × Modality × Valence).

    Bicolor markers are not supported in 3D — those classes fall back to their
    primary fill colour. Open/filled Starved/Fed convention is preserved via
    edge linewidth.
    """
    proj_s = projections['state']
    proj_m = projections['modality']
    proj_v = projections['valence']

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, cname in enumerate(class_names):
        is_starved = '(S)' in cname or cname == 'Starved'
        lw = 2.0 if is_starved else 0.5
        ax.scatter(
            float(proj_s[i]), float(proj_m[i]), float(proj_v[i]),
            c=colors[i], edgecolors=edges[i],
            s=150, linewidth=lw, depthshade=False, zorder=3,
        )

    ax.set_xlabel('State',    fontsize=FONT_SIZES['label'], labelpad=8)
    ax.set_ylabel('Modality', fontsize=FONT_SIZES['label'], labelpad=8)
    ax.set_zlabel('Valence',  fontsize=FONT_SIZES['label'], labelpad=8)
    ax.tick_params(labelsize=FONT_SIZES['tick'])

    # Draw reference planes through origin
    lim = max(
        float(np.abs(proj_s).max()),
        float(np.abs(proj_m).max()),
        float(np.abs(proj_v).max()),
    ) * 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.plot([-lim, lim], [0, 0], [0, 0], '-', color='gray', lw=0.8, alpha=0.4)
    ax.plot([0, 0], [-lim, lim], [0, 0], '-', color='gray', lw=0.8, alpha=0.4)
    ax.plot([0, 0], [0, 0], [-lim, lim], '-', color='gray', lw=0.8, alpha=0.4)

    def _update(frame):
        ax.view_init(elev=20, azim=frame * (360 / n_frames))
        return []

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=1000 // fps)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved 3D animation: {out_path}")


def main():
    styles, TASK_CLASS_NAMES, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES, TASK_BICOLOR_INFO = \
        get_style(style="styles")

    results_dict = load_all_results(BASE_RESULTS_DIR, TASK_CLASS_NAMES,
                                    fixed_trf_for_E=16, fixed_cnn_for_H=16,
                                    only_cnn_dim=16)

    for task, result in results_dict.items():
        result['__class_styles__'] = build_class_styles(
            TASK_COLORS[task], TASK_EDGECOLORS[task], TASK_SHAPES[task],
            TASK_BICOLOR_INFO.get(task, {})
        )

    plot_figure_marginals(
        results_dict, styles, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES,
        bicolor_info=TASK_BICOLOR_INFO,
        out_path=OUT_PATH,
    )

    plot_figure_pairwise_2d(
        results_dict, styles, TASK_COLORS, TASK_EDGECOLORS,
        bicolor_info=TASK_BICOLOR_INFO,
        out_path='results/CombiPlots/fig_latent_pairwise_2d.pdf',
    )

    # 3D rotating animation for task iii
    task_iii = 'State_Modality_Valence_16'
    best_data = results_dict[task_iii].get('best', {})
    X = best_data.get('transformer_latent_space')
    labels = best_data.get('latent_labels')
    class_names = results_dict[task_iii]['__class_names__']
    if X is not None and labels is not None:
        projections, _, _ = _compute_biological_axes(X, labels, task_iii, class_names)
        plot_3d_rotating(
            projections=projections,
            class_names=class_names,
            colors=TASK_COLORS[task_iii],
            edges=TASK_EDGECOLORS[task_iii],
            out_path='results/CombiPlots/fig_latent_marginals_3d.gif',
        )


if __name__ == '__main__':
    main()
