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
import plotly.graph_objects as go

from src.visualization.visualize_performance import (
    plot_1d_marginal,
    plot_2d_projection,
    _compute_biological_axes,
    draw_legend_panel,
    build_class_styles,
)
from src.utils.helpers import load_all_results, get_style
from src.visualization.figure_base import apply_style, FONT_SIZES, FIGURE_WIDTH, save_figure

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
    fig = plt.figure(figsize=(FIGURE_WIDTH, 18))

    # Outer grid: 3 task groups, height proportional to subplot count (1 : 2 : 3).
    # Extra hspace between groups; inner hspace keeps plots within a group tight.
    outer = GridSpec(3, 1, figure=fig,
                     left=0.0, right=1.0,
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

    save_figure(fig, out_path)


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

    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH, 7))

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

    save_figure(fig, out_path)


def plot_3d_interactive(
    projections, class_names, colors, edges,
    out_path='results/CombiPlots/fig_latent_marginals_3d.html',
):
    """Save an interactive drag-to-rotate 3D scatter as a standalone HTML file.

    Opens in any browser — no Python required. Bicolor classes fall back to
    their primary fill colour. Starved classes shown with open markers.
    """
    proj_s = projections['state']
    proj_m = projections['modality']
    proj_v = projections['valence']

    traces = []
    for i, cname in enumerate(class_names):
        is_starved = '(S)' in cname or cname == 'Starved'
        fill_color = colors[i] if not is_starved else 'rgba(255,255,255,0)'
        border_color = edges[i]

        traces.append(go.Scatter3d(
            x=[float(proj_s[i])],
            y=[float(proj_m[i])],
            z=[float(proj_v[i])],
            mode='markers',
            name=cname,
            marker=dict(
                size=10,
                color=fill_color,
                line=dict(color=border_color, width=3 if is_starved else 1),
                opacity=0.9,
            ),
            hovertemplate=f'<b>{cname}</b><br>State: %{{x:.3f}}<br>Modality: %{{y:.3f}}<br>Valence: %{{z:.3f}}<extra></extra>',
        ))

    # Reference lines through origin
    lim = float(max(np.abs(proj_s).max(), np.abs(proj_m).max(), np.abs(proj_v).max())) * 1.2
    for xyz in [
        dict(x=[-lim, lim], y=[0, 0], z=[0, 0]),
        dict(x=[0, 0],       y=[-lim, lim], z=[0, 0]),
        dict(x=[0, 0],       y=[0, 0], z=[-lim, lim]),
    ]:
        traces.append(go.Scatter3d(
            **xyz, mode='lines', showlegend=False,
            line=dict(color='lightgray', width=2),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title='State',
            yaxis_title='Modality',
            zaxis_title='Valence',
        ),
        title='iii. State, Modality, Valence — 3D centroid projections',
        legend=dict(itemsizing='constant'),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f"Saved interactive 3D plot: {out_path}")


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
        plot_3d_interactive(
            projections=projections,
            class_names=class_names,
            colors=TASK_COLORS[task_iii],
            edges=TASK_EDGECOLORS[task_iii],
            out_path='results/CombiPlots/fig_latent_marginals_3d.html',
        )


if __name__ == '__main__':
    main()
