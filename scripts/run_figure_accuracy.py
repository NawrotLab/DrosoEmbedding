# Fig_2.py  (Run Figure 2 - TSNE, ConfMatrices, and Barplots Modularized)
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from src.visualization.visualize_preformance import plot_confusion_matrix, plot_tsne_latent

# Base results directory
BASE_RESULTS_DIR = os.path.join('results', 'finals')

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

def plot_f1_comparison(ax, rpt_ctrl, rpt_best, class_names, show_legend=False):
    """
    Plot grouped bar chart of F1-scores for control vs best, without x-ticks,
    and using only left/bottom axes.
    """
    n_classes = len(class_names)
    width = 0.35
    ind = np.arange(n_classes)
    
    f1_ctrl = [rpt_ctrl[cls]['f1-score'] for cls in class_names]
    f1_best = [rpt_best[cls]['f1-score'] for cls in class_names]
    
    ctrl_bars = ax.bar(ind - width/2, f1_ctrl, width, label='Control', color='lightgray')
    best_bars = ax.bar(ind + width/2, f1_best, width, label='Best', color='navy')
    
    ax.set_xticks(ind)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.set_ylabel('F1 Score')
    
    if show_legend:
        ax.legend()

def plot_precision_recall_comparison(ax, rpt_best, class_names, show_legend=False):
    """
    Plot grouped bar chart of best-run precision and recall only,
    with dark/light blue and simplified axes.
    """
    n_classes = len(class_names)
    width = 0.35
    ind = np.arange(n_classes)
    
    prec_best = [rpt_best[cls]['precision'] for cls in class_names]
    rec_best = [rpt_best[cls]['recall'] for cls in class_names]
    
    prec_bars = ax.bar(ind - width/2, prec_best, width, label='Precision', color='royalblue')
    rec_bars = ax.bar(ind + width/2, rec_best, width, label='Recall', color='cornflowerblue')
    
    ax.set_xticks(ind)
    ax.set_xticklabels(class_names, rotation=90, ha='right')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    
    if show_legend:
        ax.legend()

def plot_figure2(results_dict, out_path='results/CombiPlots/fig_accuracy_v2.png'):
    """Main plotting function for Figure 2."""
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    
    # Outer grid with 3 columns with width ratios 1:2:3
    outer = GridSpec(1, 3, width_ratios=[4, 5, 6], wspace=0.4)
    
    # For each task
    for idx, (task, result) in enumerate(results_dict.items()):
        # Inner grid for each task column
        inner = GridSpecFromSubplotSpec(3, 1, 
            subplot_spec=outer[idx],
            height_ratios=[1, 1, 1],
            hspace=0.2
        )
        
        # First row: Control confusion matrix
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
            axis_labeling=None
        )
        
        # Second row: Best run confusion matrix
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
            axis_labeling=None
        )
        
        # Third row: Split into F1 and PR
        inner_bottom = GridSpecFromSubplotSpec(2, 1, 
            subplot_spec=inner[2],
            wspace=0.3
        )
        
        # F1 comparison
        ax_f1 = fig.add_subplot(inner_bottom[0])
        plot_f1_comparison(
            ax_f1,
            result['ControlRun']['classification_report_dict'],
            result['embed16_BestRun']['classification_report_dict'],
            result['__class_names__'],
            show_legend=True
        )
        
        # Remove x-axis labels from F1-score plot
        ax_f1.set_xticklabels([])
        ax_f1.tick_params(axis='x', which='both', length=0)
        
        # Precision-Recall comparison
        ax_pr = fig.add_subplot(inner_bottom[1], sharex=ax_f1)
        plot_precision_recall_comparison(
            ax_pr,
            result['embed16_BestRun']['classification_report_dict'],
            result['__class_names__'],
            show_legend=True
        )
    
    # Add overall title
    plt.suptitle("Model Performance Comparison", fontsize=16)
    
    # Adjust layout and spacing
    plt.subplots_adjust(hspace=0.1, wspace=0.2)

    out_path = f'results/CombiPlots/fig_accuracy_v2.png'
    
    # Save figure
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {out_path}")

if __name__ == '__main__':
    results = load_all_results(BASE_RESULTS_DIR)
    plot_figure2(results)
