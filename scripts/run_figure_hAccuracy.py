"""
Accuracy Figure Script 

This script generates a figure with confusion matrices and performance metrics
for different experimental conditions and model runs, with tasks arranged in rows.

Column Descriptions:
- Column 1: Confusion matrix for control model
- Column 2: Confusion matrix for best model
- Column 3: F1-score comparison between control and best models
- Column 4: Precision and recall for the best model
"""

import os
import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from src.visualization.visualize_performance import plot_confusion_matrix, plot_f1_comparison, plot_precision_recall_comparison, get_class_style, plot_model_stats, draw_legend_panel, build_class_styles
from src.utils.helpers import load_all_results, get_style, load_h16_classification_reports
from src.visualization.figure_base import apply_style, FONT_SIZES, PAGE_WIDTH

apply_style()

# Base results directory
# BASE_RESULTS_DIR = os.path.join('results', 'finals')
BASE_RESULTS_DIR = os.path.join('results', '_chkpt_finals')


def plot_figure_accuracy(results_dict, styles, out_path='results/CombiPlots/fig_accuracy_v7.png', use_class_symbols=True):
    """Main plotting function for the horizontal accuracy figure."""
    
    # Create figure with 4 rows (colorbar + 3 tasks) and appropriate columns
    fig = plt.figure(figsize=(18, 16))
    
    # Main grid for the tasks (3 rows, 3 columns)
    # Adjust the height to make space for the colorbar and titles
    outer = GridSpec(3, 1, hspace=0.1, height_ratios=[1, 1, 1], top=0.85, bottom=0.17)

    # Add column titles at the top
    column_titles = ['a. Control', 'b. Model', 'c. F1, Precision, Recall']
    col_positions = [0.2, 0.5, 0.8]  # X-positions for each column
    for col_pos, title in zip(col_positions, column_titles):
        fig.text(col_pos, 0.93, title, ha='center', va='center', fontsize=FONT_SIZES['title'], weight='bold',
        transform=fig.transFigure)
    
    # Add colorbar below the titles
    colorbar_gs = GridSpec(1, 1, top=0.9, bottom=0.88, left=0.1, right=0.59)
    #colorbar_gs = GridSpec(1, 1, top=0.1, bottom=0.08, left=0.1, right=0.59)

    cbar_ax = fig.add_subplot(colorbar_gs[0])
    # Create a colorbar matching the confusion matrix style (Blues colormap, 0-100%)
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Prediction Percentage', labelpad=10, fontsize=FONT_SIZES['colorbar'])
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.label.set_ha('right')
    cbar.ax.xaxis.label.set_x(1.0)  # still in [0,1] axes coords
    cbar.set_ticks([0, 25, 50, 75, 100])  # Explicitly set ticks at 25% intervals

    # --- Add vertical lines & labels ---
    # positions = [100/2, 100/6, 100/16] 
    positions = [1/2, 1/6, 1/16] 
    labels = ['i', 'ii', 'iii']

    trans = cbar.ax.get_xaxis_transform()

    for v, lab in zip(positions, labels):
        v = v*100
        cbar.ax.axvline(v, color='black', linestyle='--', linewidth=1)
        cbar.ax.text(v, 1.2, lab, transform=trans, ha='center', va='bottom',
                    fontsize=FONT_SIZES['colorbar'], fontweight='bold')
        
    # Add row names on the left
    row_titles = {
        'MetabolicState_2': 'i. State',
        'State_Modality_6': 'ii. State, Modality',
        'State_Modality_Valence_16': 'iii. State, Modality, Valence'
    }
    
    # Styles are already loaded at the beginning of the function
    
    # Store reference to first ax_stats for sharing x-axis
    first_ax_stats = None

    
    
    # For each task (now in rows)
    for row_idx, (task, result) in enumerate(results_dict.items()):

        # Add row title
        row_y = 0.8 - (row_idx * 0.3)  # Adjust vertical position based on row index
        row_y = [0.75, 0.52, 0.28]
        fig.text(0.05, row_y[row_idx], row_titles[task],
                ha='left', va='center', fontsize=FONT_SIZES['title'],
                transform=fig.transFigure, rotation=90, weight='bold')
        # Inner grid for each task row with 3 columns
        inner = GridSpecFromSubplotSpec(1, 3, 
            subplot_spec=outer[row_idx],
            width_ratios=[1, 1, 1.5],
            wspace=0.15
        )
        
        # First column: Control confusion matrix
        ax_ctrl = fig.add_subplot(inner[0])
        plot_confusion_matrix(
            cl_name=f"Control {task}",
            cm=result['control']['confusion_matrix'],
            class_names=result['__class_names__'],
            output_path=None,
            dataID=task,
            ax=ax_ctrl,
            annot=False,
            cbar=False, 
            axis_labeling='y_axis',
            use_class_symbols=use_class_symbols,
            styles=styles,
            class_styles=result.get('__class_styles__')
        )
        
        # Second column: Best run confusion matrix
        ax_best = fig.add_subplot(inner[1])
        plot_confusion_matrix(
            cl_name=f"Best {task}",
            cm=result['best']['confusion_matrix'],
            class_names=result['__class_names__'],
            output_path=None,
            dataID=task,
            ax=ax_best,
            annot=False,
            cbar=False, 
            axis_labeling=None,
            use_class_symbols=use_class_symbols,
            styles=styles,
            class_styles=result.get('__class_styles__')
        )
        # ax_best.set_title('Best Run', fontsize=10)
        
        # Third column: Model stats plot
        # Share x-axis with first row if it exists
        if first_ax_stats is None:
            ax_stats = fig.add_subplot(inner[2])
            first_ax_stats = ax_stats
        else:
            ax_stats = fig.add_subplot(inner[2], sharex=first_ax_stats)
        
        all_reports = result.get('__h16_reports__') or load_h16_classification_reports(result)
        # Plot model stats (F1, Precision, Recall for both control and model)
        plot_model_stats(
            ax_stats,
            result['control']['classification_report_dict'],
            result['best']['classification_report_dict'],
            result['__class_names__'],
            rpt_all_runs=all_reports,
            show_legend=False,
            use_class_symbols=use_class_symbols,
            styles=styles,
            class_styles=result.get('__class_styles__')
        )
        
        # Hide x-axis labels for top two rows, only show on bottom row
        if row_idx < 2:  # Top two rows
            plt.setp(ax_stats.get_xticklabels(), visible=False)
    
    # Draw shared legend panel at bottom of figure
    draw_legend_panel(fig, styles, line_y=0.13,
                      ax_rect=[0.03, 0.02, 0.95, 0.095])
    
    # Adjust layout to accommodate the colorbar and legends
    plt.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.17)
    
    # Save figure
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.svg'), dpi=300, format='svg', bbox_inches='tight', pad_inches=0.15)
    plt.savefig(out_path.replace('.png', '.pdf'), dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"Figure saved to {out_path}")

def main():
    print(f"Python: {sys.executable}")
    """Main function to load results and generate the figure."""
    # Load styles and class names from shared config
    styles, TASK_CLASS_NAMES, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES, TASK_BICOLOR_INFO = get_style(style="styles")

    # Load results — only_cnn_dim=16 skips unneeded dim combos (much faster)
    results = load_all_results(BASE_RESULTS_DIR, TASK_CLASS_NAMES,
                               fixed_trf_for_E=16, fixed_cnn_for_H=16,
                               only_cnn_dim=16)

    for task, result in results.items():
        # Load H16 reports once and cache on the result dict
        all_reports = load_h16_classification_reports(result)
        result['__h16_reports__'] = all_reports

        # Build per-class style dicts (handles bicolor entries correctly)
        result['__class_styles__'] = build_class_styles(
            TASK_COLORS[task], TASK_EDGECOLORS[task], TASK_SHAPES[task],
            TASK_BICOLOR_INFO.get(task, {})
        )

        accs = [rpt['accuracy'] for rpt in all_reports]
        ctrl_acc = result['control']['classification_report_dict']['accuracy']
        best_acc = result['best']['classification_report_dict']['accuracy']
        print(f"\n{task}")
        print(f"  Control accuracy:       {ctrl_acc:.3f}")
        print(f"  Best run accuracy:      {best_acc:.3f}")
        print(f"  Mean acc (50 runs):     {np.mean(accs):.3f}")
        print(f"  Median acc (50 runs):   {np.median(accs):.3f}")
        print(f"  IQR (25-75):            {np.percentile(accs,25):.3f} -- {np.percentile(accs,75):.3f}")
        # Identify worst classes in task iii
        if '16' in task:
            best_rpt = result['best']['classification_report_dict']
            class_names = result['__class_names__']
            f1s = [(name, best_rpt[name]['f1-score']) for name in class_names]
            f1s.sort(key=lambda x: x[1])
            print(f"  3 lowest F1 classes:    {f1s[:3]}")
    
    
    # Generate and save the figure
    plot_figure_accuracy(results, styles)

if __name__ == '__main__':
    main()