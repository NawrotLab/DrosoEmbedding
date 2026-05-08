"""
Latent Space Visualization Script

This script generates a figure with t-SNE visualizations and centroid vectors
for different experimental conditions and model runs.

Plotting helpers live in src.visualization.visualize_performance;
this script only handles figure layout + assembly.
"""

import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.visualization.visualize_performance import (
    draw_legend_panel,
    plot_biological_axes_panel,
    plot_accuracy_vs_dimension,
    plot_tsne_panel,
)
from src.utils.helpers import load_all_results, get_style
from src.utils.logger import setup_logger
from src.visualization.figure_base import apply_style, FONT_SIZES, PAGE_WIDTH, FigureConfig

apply_style()


# ──────────────────────────────────────────────
# Figure layout
# ──────────────────────────────────────────────

def _setup_figure() -> Tuple[plt.Figure, GridSpec]:
    """Set up the figure and grid layout."""

    fig = plt.figure(figsize=(18, 17))

    gs = GridSpec(3, 4, figure=fig,
                 left=0.08, right=0.98,
                 bottom=0.20, top=0.93,
                 wspace=0.15, hspace=0.25,
                 width_ratios=[1, 1, 1, 0.8])

    return fig, gs


# ──────────────────────────────────────────────
# Main figure function
# ──────────────────────────────────────────────

def plot_figure_latent(
    results_dict: Dict,
    styles: Dict,
    colors: Dict,
    edges: Dict,
    shapes: Dict,
    bicolor_info: Dict = None,
    out_path: str = 'results/CombiPlots/fig_latent_chptRuns_1.png',
    use_l_axis: bool = True,
    logger=None,
) -> None:
    """Generate the complete figure with t-SNE plots and accuracy vs dimension."""
    fig, gs = _setup_figure()

    tasks = list(results_dict.keys())[:3]
    task_names = ['i. State', 'ii. State, Modality', 'iii. State, Modality, Valence']
    task_y_pos = [0.81, 0.53, 0.25]

    column_titles = ['a. Control t-SNE', 'b. Model t-SNE', 'c. Centroid projections', 'd. Accuracy']
    column_positions = [0.18, 0.43, 0.65, 0.9]

    for x, title in zip(column_positions, column_titles):
        fig.text(x, 0.95, title, ha='center', va='center', fontsize=FONT_SIZES['title'], weight='bold')

    # Plot each task
    for row, (task, task_name) in enumerate(zip(tasks, task_names)):
        if logger:
            logger.debug(f"Plotting task: {task} (row {row})")
        run_dict = results_dict[task]
        class_names = run_dict.get('__class_names__')
        if logger:
            logger.debug(f"Task '{task}': class_names = {class_names}")

        # Row label
        fig.text(0.05, task_y_pos[row], task_name, ha='left', va='center',
                 fontsize=FONT_SIZES['title'], rotation=90, weight='bold', transform=fig.transFigure)

        task_bicolor = bicolor_info.get(task, {}) if bicolor_info else {}

        # Column 0 — Control t-SNE
        plot_tsne_panel(fig, gs, row, 0, run_dict, class_names,
                        colors[task], edges[task], shapes[task],
                        bicolor_info=task_bicolor, use_l_axis=use_l_axis,
                        is_control=True, task=task, logger=logger)

        # Column 1 — Best t-SNE
        plot_tsne_panel(fig, gs, row, 1, run_dict, class_names,
                        colors[task], edges[task], shapes[task],
                        bicolor_info=task_bicolor, use_l_axis=use_l_axis,
                        task=task, logger=logger)

        # Column 2 — Biological axes (full high-D latent space)
        best_data = run_dict.get('best', {})
        X_hd = best_data.get('transformer_latent_space')
        latent_labels = best_data.get('latent_labels')

        ax_bio = fig.add_subplot(gs[row, 2])
        if X_hd is not None and latent_labels is not None:
            plot_biological_axes_panel(
                ax=ax_bio,
                X=X_hd,
                labels=latent_labels,
                class_names=class_names,
                task=task,
                colors=colors[task],
                edges=edges[task],
                bicolor_info=task_bicolor,
                fig=fig,
                gs=gs,
                row=row,
            )
        else:
            ax_bio.set_xticks([]); ax_bio.set_yticks([])
            if logger:
                logger.warning(f"Task '{task}': missing transformer_latent_space for biological axes")

        # Column 3 — Accuracy vs dimension
        ax = fig.add_subplot(gs[row, 3])
        primary_family = "H"
        best_dim = run_dict.get('best_trf_dim') if primary_family == "H" else run_dict.get('best_cnn_dim')
        plot_accuracy_vs_dimension(ax, run_dict, task_name,
                                   color=list(colors.values())[row], row=row,
                                   primary_family=primary_family,
                                   overlay_alt_family=False, best_dim=best_dim,
                                   baseline=FigureConfig.TASK_CONFIG.get(task, {}).get('baseline'))

    # Draw legend panel at bottom of figure
    draw_legend_panel(fig, styles, line_y=0.165)

    # Save figure
    if logger:
        logger.debug(f"Saving figure to {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.savefig(out_path.replace('.png', '.svg'), dpi=300, format='svg', bbox_inches='tight', pad_inches=0.15)
    plt.savefig(out_path.replace('.png', '.pdf'), dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.15)
    if logger:
        logger.info(f"Saved SVG : {out_path.replace('.png', '.svg')}")
        logger.info(f"Saved PDF : {out_path.replace('.png', '.pdf')}")
    plt.close()
    if logger:
        logger.info(f"Figure saved to {out_path}")
    else:
        print(f"Figure saved to {out_path}")


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main():
    logger = setup_logger(task_name="run_figure_hLatent", log_dir="logs/run_figure_hLatent")
    logger.info("Starting main")

    BASE_RESULTS_DIR = os.path.join('results', '_chkpt_finals')
    logger.info(f"BASE_RESULTS_DIR: {BASE_RESULTS_DIR}")
    logger.debug(f"Checking if BASE_RESULTS_DIR exists: {os.path.exists(BASE_RESULTS_DIR)}")

    logger.debug("Loading styles from YAML file...")
    styles, TASK_CLASS_NAMES, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES, TASK_BICOLOR_INFO = get_style(style="styles")
    logger.debug(f"Loaded styles. TASK_CLASS_NAMES keys: {list(TASK_CLASS_NAMES.keys()) if isinstance(TASK_CLASS_NAMES, dict) else 'N/A'}")

    logger.debug(f"Loading results from {BASE_RESULTS_DIR}...")
    results_dict = load_all_results(BASE_RESULTS_DIR, TASK_CLASS_NAMES,
                                     fixed_trf_for_E=16, fixed_cnn_for_H=16,
                                     only_cnn_dim=16, logger=logger)
    logger.debug(f"Loaded results. Tasks found: {list(results_dict.keys())}")

    # Debug: Check what was loaded for each task
    for task_name, task_data in results_dict.items():
        logger.debug(f"Task '{task_name}':")
        logger.debug(f"  - Has control: {task_data.get('control') is not None}")
        logger.debug(f"  - Has best: {task_data.get('best') is not None}")
        logger.debug(f"  - Run groups: {list(task_data.get('runs', {}).keys())}")
        for group_key, runs in task_data.get('runs', {}).items():
            logger.debug(f"    - Group {group_key}: {len(runs)} runs")

    logger.info("Loaded results successfully")

    logger.debug("Generating figure...")
    plot_figure_latent(results_dict, styles, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES,
                       bicolor_info=TASK_BICOLOR_INFO,
                       out_path='results/CombiPlots/fig_Latent_v13.png', logger=logger)
    logger.info("Figure generation complete")


if __name__ == '__main__':
    main()