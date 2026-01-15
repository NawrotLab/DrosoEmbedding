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
import yaml
import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from src.visualization.visualize_preformance import plot_confusion_matrix, plot_f1_comparison, plot_precision_recall_comparison, get_class_style, plot_model_stats
from src.utils.helpers import load_all_results, get_style


# Base results directory
# BASE_RESULTS_DIR = os.path.join('results', 'finals')
BASE_RESULTS_DIR = os.path.join('results', '_chkpt_finals')


# Task configurations
TASK_CLASS_NAMES = {
    'MetabolicState_2': ["Starved", "Fed"],
    'State_Modality_6': [
        "Odor (S)", "Odor (F)", "Taste (S)", "Taste (F)",
        "Odor + Taste (S)", "Odor + Taste (F)"
    ],
    'State_Modality_Valence_16': [
        "O$^{+}$ (S)", "O$^{-}$ (S)", "O$^{+}$ (F)", "O$^{-}$ (F)",
        "T$^{+}$ (S)", "T$^{-}$ (S)", "T$^{+}$ (F)", "T$^{-}$ (F)",
        "O$^{+}$+T$^{+}$ (S)", "O$^{-}$+T$^{-}$ (S)", "O$^{-}$+T$^{+}$ (S)", "O$^{+}$+T$^{-}$ (S)",
        "O$^{+}$+T$^{+}$ (F)", "O$^{-}$+T$^{-}$ (F)", "O$^{-}$+T$^{+}$ (F)", "O$^{+}$+T$^{-}$ (F)"
    ]
}


def load_styles():
    """Load styles from the stylesE.yaml file."""
    styles_path = os.path.join('src', 'visualization', 'stylesE.yaml')
    with open(styles_path, 'r') as f:
        styles = yaml.safe_load(f)['styles']
    return styles

def plot_figure_accuracy(results_dict, out_path='results/CombiPlots/fig_accuracy_horizontal_chkpt_v4.png', use_class_symbols=True):
    """Main plotting function for the horizontal accuracy figure."""
    # Set up figure with styles
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    
    # Load styles if needed
    styles = load_styles() if use_class_symbols else None
    
    # Create figure with 4 rows (colorbar + 3 tasks) and appropriate columns
    fig = plt.figure(figsize=(18, 16))
    
    # Main grid for the tasks (3 rows, 3 columns)
    # Adjust the height to make space for the colorbar and titles
    outer = GridSpec(3, 1, hspace=0.1, height_ratios=[1, 1, 1], top=0.85, bottom=0.05)

    # Add column titles at the top
    column_titles = ['a Control', 'b Model', 'c F1, Precision, Recall']
    col_positions = [0.2, 0.5, 0.8]  # X-positions for each column
    for col_pos, title in zip(col_positions, column_titles):
        fig.text(col_pos, 0.93, title, ha='center', va='center', fontsize=16, weight='bold', 
        transform=fig.transFigure)
    
    # Add colorbar below the titles
    colorbar_gs = GridSpec(1, 1, top=0.9, bottom=0.88, left=0.1, right=0.59)
    #colorbar_gs = GridSpec(1, 1, top=0.1, bottom=0.08, left=0.1, right=0.59)

    cbar_ax = fig.add_subplot(colorbar_gs[0])
    # Create a colorbar matching the confusion matrix style (Blues colormap, 0-100%)
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Prediction Percentage', labelpad=10, fontsize=10)
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
                    fontsize=9, fontweight='bold')
        
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
        row_y = [0.75, 0.45, 0.16]
        fig.text(0.05, row_y[row_idx], row_titles[task], 
                ha='left', va='center', fontsize=16, #weight='bold', 
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
            axis_labeling='y_axis',  # Changed from 'x_axis' to 'both' to show both axes
            use_class_symbols=use_class_symbols,
            styles=styles
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
            axis_labeling=None,  # Changed from 'x_axis' to 'both' to show both axes
            use_class_symbols=use_class_symbols,
            styles=styles
        )
        # ax_best.set_title('Best Run', fontsize=10)
        
        # Third column: Model stats plot
        # Share x-axis with first row if it exists
        if first_ax_stats is None:
            ax_stats = fig.add_subplot(inner[2])
            first_ax_stats = ax_stats
        else:
            ax_stats = fig.add_subplot(inner[2], sharex=first_ax_stats)
        
        # Plot model stats (F1, Precision, Recall for both control and model)
        plot_model_stats(
            ax_stats,
            result['control']['classification_report_dict'],
            result['best']['classification_report_dict'],
            result['__class_names__'],
            show_legend=False,  # Legend will be added separately at the top
            use_class_symbols=use_class_symbols,
            styles=styles
        )
        
        # Hide x-axis labels for top two rows, only show on bottom row
        if row_idx < 2:  # Top two rows
            plt.setp(ax_stats.get_xticklabels(), visible=False)
    
    # Add legend at the top, aligned with colorbar
    # Create legend handles manually (matching plot_model_stats style)
    from matplotlib.patches import Rectangle
    ctrl_handle = Rectangle((0, 0), 1, 1, fill=False, edgecolor='#b7bec4', linewidth=2)
    model_handle = Rectangle((0, 0), 1, 1, fill=False, edgecolor='#094c80', linewidth=2)
    handles = [ctrl_handle, model_handle]
    labels = ['Control', 'Model']
    
    legend_gs = GridSpec(1, 1, top=0.9, bottom=0.88, left=0.6, right=0.98)
    legend_ax = fig.add_subplot(legend_gs[0])
    legend_ax.axis('off')
    legend = legend_ax.legend(handles, labels, 
                             loc='center', 
                             ncol=len(labels),
                             fontsize=15, 
                             frameon=False)
    legend.set_in_layout(False)
    
    # Adjust layout to accommodate the colorbar and legends
    plt.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.1)
    
    # Save figure
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {out_path}")

def main():
    print(f"Python: {sys.executable}")
    """Main function to load results and generate the figure."""
    # Load results
    results = load_all_results(BASE_RESULTS_DIR, TASK_CLASS_NAMES)
    
    # Generate and save the figure
    plot_figure_accuracy(results)

if __name__ == '__main__':
    main()
