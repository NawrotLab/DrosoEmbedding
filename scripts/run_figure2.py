# Fig_2.py  (Run Figure 2 - TSNE, ConfMatrices, and Barplots Modularized)
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from src.visualization.visualize_preformance import plot_confusion_matrix, plot_tsne_latent

# Base results directory
BASE_RESULTS_DIR = os.path.join('results', 'finals')
# Columns to plot
COLUMNS = ['tsne_ctrl', 'tsne_best', 'xx', 'conf_matrix', 'yy']
# Task-specific class names
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
    """Load eval pickles plus attach class names."""
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

def plot_centroid_vectors(latent, labels, class_names, task, ax):
    """
    Plot vectors from origin to class-group centroids based on task-specific groupings.
    """
    # Define groupings
    if task == 'MetabolicState_2':
        group_map = {
            'Starved': [0],
            'Fed': [1]
        }
    elif task == 'State_Modality_6':
        group_map = {
            'Starved': [0, 2, 4],
            'Fed': [1, 3, 5],
            'Odor': [0, 1],
            'Taste': [2, 3],
            'Odor+Taste': [4, 5]
        }
    elif task == 'State_Modality_Valence_16':
        # Build Pos/Neg based on name matching
        pos_inds = [i for i, n in enumerate(class_names) if '+' in n and '-' not in n]
        neg_inds = [i for i, n in enumerate(class_names) if '-' in n and '+' not in n]
        group_map = {
            'Starved': [0,1,4,5,8,9,10,11],
            'Fed': [2,3,6,7,12,13,14,15],
            
            'Odor': [0, 1, 2, 3],
            'Taste': [4, 5, 6, 7],
            'Odor+Taste': [8, 9, 10,11,12,13],

            'Positive': [0,2,4,6,8,12],
            'Negative': [1,3,5,7,9,13],
            'Valence-Combi': [10, 11, 14, 15]
        }
    else:
        group_map = {}

        # Plot arrows with grouped colors
    # Define colors
    col_starve = 'tab:purple'
    col_sensory = 'tab:green'
    col_valence = 'tab:red'
    for name, inds in group_map.items():
        mask = np.isin(labels, inds)
        if not np.any(mask):
            continue
        centroid = latent[mask].mean(axis=0)
        # choose color group
        if name in ['Starved', 'Fed']:
            col = col_starve
        elif name in ['Odor', 'Taste', 'Odor+Taste']:
            col = col_sensory
        else:
            col = col_valence
        ax.annotate(
            '', xy=(centroid[0], centroid[1]), xytext=(0,0),
            arrowprops=dict(arrowstyle='->', linewidth=1.5, color=col)
        )
        ax.text(
            centroid[0]*1.1, centroid[1]*1.1, name,
            fontsize=8, ha='center', va='center', color=col
        )

    # Axes and formatting: show only central lines, no box or titles, fixed limits
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_aspect('equal', 'box')
    # show only the endpoints on ticks
    ax.set_xticks([-100, 100])
    ax.set_yticks([-100, 100])

def plot_f1_comparison(ax, rpt_ctrl, rpt_best, class_names, show_legend=False):
    """
    Plot grouped bar chart of F1-scores for control vs best, without x-ticks,
    and using only left/bottom axes.
    """
    inds = np.arange(len(class_names))
    width = 0.35
    f1_ctrl = [rpt_ctrl[c]['f1-score']*100 for c in class_names]
    f1_best = [rpt_best[c]['f1-score']*100 for c in class_names]
    ax.bar(inds - width/2, f1_ctrl, width, label='Ctrl F1', color='lightgray')
    ax.bar(inds + width/2, f1_best, width, label='Best F1', color='steelblue')
    # remove x-ticks
    ax.set_xticks([])
    # simplify spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_ylabel('F1 (%)', fontsize=10)
    ax.set_ylim([0, 100])
    if show_legend:
        ax.legend(fontsize=8)

def plot_precision_recall_comparison(ax, rpt_ctrl, rpt_best, class_names, show_legend=False):
    """
    Plot grouped bar chart of best-run precision and recall only,
    with dark/light blue and simplified axes.
    """
    inds = np.arange(len(class_names))
    width = 0.4
    prec_best = [rpt_best[c]['precision']*100 for c in class_names]
    rec_best  = [rpt_best[c]['recall']*100    for c in class_names]
    offsets = np.array([-0.5, 0.5]) * width
    metrics = [prec_best, rec_best]
    colors  = ['navy', 'skyblue']
    labels  = ['Best P', 'Best R']
    for k, vals in enumerate(metrics):
        ax.bar(inds + offsets[k], vals, width, label=labels[k], color=colors[k])
    ax.set_xticks(inds)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    # simplify spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_ylabel('%', fontsize=10)
    if show_legend:
        ax.legend(fontsize=8, ncol=2)

def plot_figure2(results_dict, out_path='results/CombiPlots/fig2.png'):
    tasks = list(results_dict.keys())
    n_tasks = len(tasks)
    n_cols = len(COLUMNS)
    gs = GridSpec(n_tasks, n_cols, wspace=0.3, hspace=0.4)
    fig = plt.figure(figsize=(n_cols * 4, n_tasks * 4))

    for i, (task, run_dict) in enumerate(results_dict.items()):
        class_names = run_dict['__class_names__']
        ctrl = 'control' if 'control' in run_dict else next(k for k in run_dict if k not in ('__class_names__','embed16_BestRun'))
        best = 'embed16_BestRun' if 'embed16_BestRun' in run_dict else ctrl
        ctrl_tsne = run_dict[ctrl]['tsne_2d']
        ctrl_labels = run_dict[ctrl]['latent_labels']
        best_tsne = run_dict[best]['tsne_2d']
        best_labels = run_dict[best]['latent_labels']
        cm_best = run_dict[best]['confusion_matrix']
        rpt_ctrl = run_dict[ctrl]['classification_report_dict']
        rpt_best = run_dict[best]['classification_report_dict']

        for j, col in enumerate(COLUMNS):
            if col == 'tsne_ctrl':
                ax = fig.add_subplot(gs[i, j])
                plot_tsne_latent(latent_2d=ctrl_tsne, labels_np=ctrl_labels,
                                 class_names=class_names, ax=ax,
                                 legend=False, draw_axis=False, draw_title=False)
            elif col == 'tsne_best':
                ax = fig.add_subplot(gs[i, j])
                plot_tsne_latent(latent_2d=best_tsne, labels_np=best_labels,
                                 class_names=class_names, ax=ax,
                                 legend=False, draw_axis=False, draw_title=False)
                
            elif col == 'xx':
                ax = fig.add_subplot(gs[i, j])
                plot_centroid_vectors(best_tsne, best_labels, class_names, task, ax)
                
            elif col == 'conf_matrix':
                ax = fig.add_subplot(gs[i, j])
                do_annot = i < (n_tasks - 1)
                plot_confusion_matrix(cl_name=best, cm=cm_best,
                                      class_names=class_names, output_path=None,
                                      dataID=task, hyperparameters=None,
                                      ax=ax, annot=do_annot, cbar=False)
            elif col == 'yy':
                sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i, j], height_ratios=[1, 1], hspace=0.3)
                ax1 = fig.add_subplot(sub[0])
                plot_f1_comparison(ax1, rpt_ctrl, rpt_best, class_names, show_legend=(i==0))
                ax2 = fig.add_subplot(sub[1])
                plot_precision_recall_comparison(ax2, rpt_ctrl, rpt_best, class_names, show_legend=(i==0))
            else:
                ax = fig.add_subplot(gs[i, j])
                ax.axis('off')

            if j == 0 and col != 'yy':
                ax.set_ylabel(task, fontsize=14, rotation=90, labelpad=20)

    fig.text(0.25, 0.97, 'Control', ha='center', fontsize=16)
    fig.text(0.75, 0.97, 'Best Run', ha='center', fontsize=16)
    fig.text(0.25, 0.02, 't-SNE Component 1', ha='center', fontsize=14)
    fig.text(0.75, 0.02, 't-SNE Component 1', ha='center', fontsize=14)
    top_margin = 0.1; mid_h = 0.8
    for idx in range(n_tasks):
        y = top_margin + (n_tasks - idx - 0.5)*(mid_h/n_tasks)
        fig.text(0.05, y, 't-SNE Component 2', va='center', rotation='vertical', fontsize=14)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved to {out_path}")

if __name__ == '__main__':
    results = load_all_results(BASE_RESULTS_DIR)
    plot_figure2(results)
