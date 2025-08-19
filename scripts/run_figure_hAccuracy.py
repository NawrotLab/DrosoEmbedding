"""
Accuracy Figure Script (Horizontal Layout)

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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from src.visualization.visualize_preformance import plot_confusion_matrix, plot_f1_comparison, plot_precision_recall_comparison, get_class_style

# Base results directory
BASE_RESULTS_DIR = os.path.join('results', 'finals')

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

def load_all_results(base_dir):
    """Load all results from the base directory."""
    results = {}
    for task in sorted(os.listdir(base_dir)):
        task_path = os.path.join(base_dir, task)
        if not os.path.isdir(task_path):
            continue
        results[task] = {}
        for run_name in sorted(os.listdir(task_path)):
            pkl = os.path.join(task_path, run_name, 'evaluation_results.pkl')
            if os.path.isfile(pkl):
                with open(pkl, 'rb') as f:
                    results[task][run_name] = pickle.load(f)
        if task in TASK_CLASS_NAMES:
            results[task]['__class_names__'] = TASK_CLASS_NAMES[task]
    return results

def load_styles():
    """Load styles from the stylesE.yaml file."""
    styles_path = os.path.join('src', 'visualization', 'stylesE.yaml')
    with open(styles_path, 'r') as f:
        styles = yaml.safe_load(f)['styles']
    return styles

def plot_figure_accuracy(results_dict, out_path='results/CombiPlots/fig_accuracy_horizontal_v5.png', use_class_symbols=True):
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
    column_titles = ['(a) Control', '(b) Model', '(c) F1, Precision, Recall']
    col_positions = [0.2, 0.5, 0.8]  # X-positions for each column
    for col_pos, title in zip(col_positions, column_titles):
        fig.text(col_pos, 0.93, title, ha='center', va='center', fontsize=16, #weight='bold', 
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
    cbar.set_ticks([0, 25, 50, 75, 100])  # Explicitly set ticks at 25% intervals
    
    # Add row names on the left
    row_titles = {
        'MetabolicState_2': 'i. State',
        'State_Modality_6': 'ii. State, Modality',
        'State_Modality_Valence_16': 'iii. State, Modality, Valence'
    }
    
    # Styles are already loaded at the beginning of the function
    
    # For each task (now in rows)
    for row_idx, (task, result) in enumerate(results_dict.items()):
        # Add row title
        row_y = 0.8 - (row_idx * 0.3)  # Adjust vertical position based on row index
        row_y = [0.75, 0.45, 0.16]
        fig.text(0.05, row_y[row_idx], row_titles[task], 
                ha='left', va='center', fontsize=16, #weight='bold', 
                transform=fig.transFigure, rotation=90)
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
            cm=result['ControlRun']['confusion_matrix'],
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
            cm=result['embed16_BestRun']['confusion_matrix'],
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
        
        # Third column: Split into F1 and PR
        inner_right = GridSpecFromSubplotSpec(2, 1, 
            subplot_spec=inner[2],
            hspace=0.07
        )
        
        # Create shared axes for F1 and PR plots
        ax_f1 = fig.add_subplot(inner_right[0])
        ax_pr = fig.add_subplot(inner_right[1], sharex=ax_f1)
        
        # Turn off x-tick labels for the top plot (F1)
        plt.setp(ax_f1.get_xticklabels(), visible=False)
        
        # Plot F1 comparison (top)
        plot_f1_comparison(
            ax_f1, 
            result['ControlRun']['classification_report_dict'], 
            result['embed16_BestRun']['classification_report_dict'],
            result['__class_names__'],
            show_legend=False,
            use_class_symbols=use_class_symbols,
            styles=None
        )
        
        # Plot precision/recall comparison (bottom)
        plot_precision_recall_comparison(
            ax_pr,
            result['embed16_BestRun']['classification_report_dict'],
            result['__class_names__'],
            show_legend=False, 
            use_class_symbols=use_class_symbols,
            styles=styles
        )
        # ax_pr.set_title('Precision & Recall', fontsize=10)
        
        # Add class names only to the bottom plot with rotation for better readability
        # ax_pr.set_xticklabels(result['__class_names__'], rotation=45, ha='right')
        
            # Adjust spacing between subplots
        # plt.subplots_adjust(hspace=0.01)
        
        # Move legend to the last column for better visibility
        if row_idx == 0:  # Only add legend to the first row's plots
            # Get handles and labels from both F1 and PR plots
            handles_f1, labels_f1 = ax_f1.get_legend_handles_labels()
            handles_pr, labels_pr = ax_pr.get_legend_handles_labels()
            
            # Combine handles and labels, removing duplicates
            handles = handles_f1 + [h for h in handles_pr if h not in handles_f1]
            labels = labels_f1 + [l for l in labels_pr if l not in labels_f1]
            
            # Create a new subplot for the legend using figure coordinates
            # Convert figure coordinates to axes coordinates
            legend_ax = fig.add_axes([0.66, 0.84, 0.3, 0.1])  # [left, bottom, width, height] in figure coordinates
            legend_ax.axis('off')  # Hide the axes
            
            # Add legend to the new subplot
            legend = legend_ax.legend(handles, labels, 
                                   loc='center', 
                                   ncol=len(labels),
                                   fontsize=13, 
                                   frameon=False)
            legend.set_in_layout(False)  # Prevent layout adjustments
    
    # Adjust layout to accommodate the colorbar and legends
    plt.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.1)
    
    # Save figure
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {out_path}")

def main():
    """Main function to load results and generate the figure."""
    # Load results
    results = load_all_results(BASE_RESULTS_DIR)
    
    # Generate and save the figure
    plot_figure_accuracy(results)

if __name__ == '__main__':
    main()
