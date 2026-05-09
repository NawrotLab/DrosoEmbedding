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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from src.visualization.visualize_performance import (
    plot_1d_marginal,
    _compute_biological_axes,
    draw_legend_panel,
    build_class_styles,
)
from src.utils.helpers import load_all_results, get_style
from src.visualization.figure_base import apply_style, FONT_SIZES

apply_style()

BASE_RESULTS_DIR = os.path.join('results', '_chkpt_finals')
OUT_PATH = 'figures/fig_latent_marginals.pdf'

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
                     left=0.10, right=0.98,
                     top=0.93, bottom=0.10,
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

    draw_legend_panel(fig, styles, line_y=0.085)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


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


if __name__ == '__main__':
    main()
