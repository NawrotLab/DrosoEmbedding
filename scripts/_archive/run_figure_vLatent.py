# Fig_2.py  (Run Figure 2 - TSNE, ConfMatrices, and Barplots Modularized)
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from src.visualization.visualize_preformance import plot_confusion_matrix, plot_tsne_latent
import yaml 

with open("src/visualization/stylesD.yaml", "r") as f:
    styles = yaml.safe_load(f)["styles"]

# Base results directory
BASE_RESULTS_DIR = os.path.join('results', 'finals')
# Columns to plot
ROWS = ['tsne_ctrl', 'tsne_best', 'xx']
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
TASK_COLORS = {
    'MetabolicState_2': [styles['starved']['color'], styles['fed']['color']],
    'State_Modality_6': [
        styles['starved_odor']['color'], styles['fed_odor']['color'],
        styles['starved_taste']['color'], styles['fed_taste']['color'],
        styles['starved_odor_taste']['color'], styles['fed_odor_taste']['color']
    ],
    'State_Modality_Valence_16': [
        styles['starved_odor_positive']['color'], styles['starved_odor_negative']['color'],
        styles['starved_taste_positive']['color'], styles['starved_taste_negative']['color'],
        styles['starved_odor_pos_taste_pos']['color'], styles['starved_odor_neg_taste_neg']['color'],
        styles['starved_odor_neg_taste_pos']['color'], styles['starved_odor_pos_taste_neg']['color'],
        styles['fed_odor_positive']['color'], styles['fed_odor_negative']['color'],
        styles['fed_taste_positive']['color'], styles['fed_taste_negative']['color'],
        styles['fed_odor_pos_taste_pos']['color'], styles['fed_odor_neg_taste_neg']['color'],
        styles['fed_odor_neg_taste_pos']['color'], styles['fed_odor_pos_taste_neg']['color']
    ]
}

TASK_EDGECOLORS = {
    'MetabolicState_2': [styles['starved']['edgecolor'], styles['fed']['edgecolor']],
    'State_Modality_6': [
        styles['starved_odor']['edgecolor'], styles['fed_odor']['edgecolor'],
        styles['starved_taste']['edgecolor'], styles['fed_taste']['edgecolor'],
        styles['starved_odor_taste']['edgecolor'], styles['fed_odor_taste']['edgecolor']
    ],
    'State_Modality_Valence_16': [
        styles['starved_odor_positive']['edgecolor'], styles['starved_odor_negative']['edgecolor'],
        styles['starved_taste_positive']['edgecolor'], styles['starved_taste_negative']['edgecolor'],
        styles['starved_odor_pos_taste_pos']['edgecolor'], styles['starved_odor_neg_taste_neg']['edgecolor'],
        styles['starved_odor_neg_taste_pos']['edgecolor'], styles['starved_odor_pos_taste_neg']['edgecolor'],
        styles['fed_odor_positive']['edgecolor'], styles['fed_odor_negative']['edgecolor'],
        styles['fed_taste_positive']['edgecolor'], styles['fed_taste_negative']['edgecolor'],
        styles['fed_odor_pos_taste_pos']['edgecolor'], styles['fed_odor_neg_taste_neg']['edgecolor'],
        styles['fed_odor_neg_taste_pos']['edgecolor'], styles['fed_odor_pos_taste_neg']['edgecolor']
    ]
}

TASK_SHAPES = {
    'MetabolicState_2': [styles['starved']['shape'], styles['fed']['shape']],
    'State_Modality_6': [
        styles['starved_odor']['shape'], styles['starved_taste']['shape'],
        styles['starved_odor_taste']['shape'], styles['fed_odor']['shape'],
        styles['fed_taste']['shape'], styles['fed_odor_taste']['shape']
    ],
    'State_Modality_Valence_16': [
        styles['starved_odor_positive']['shape'], styles['starved_odor_negative']['shape'],
        styles['starved_taste_positive']['shape'], styles['starved_taste_negative']['shape'],
        styles['starved_odor_pos_taste_pos']['shape'], styles['starved_odor_neg_taste_neg']['shape'],
        styles['starved_odor_neg_taste_pos']['shape'], styles['starved_odor_pos_taste_neg']['shape'],
        styles['fed_odor_positive']['shape'], styles['fed_odor_negative']['shape'],
        styles['fed_taste_positive']['shape'], styles['fed_taste_negative']['shape'],
        styles['fed_odor_pos_taste_pos']['shape'], styles['fed_odor_neg_taste_neg']['shape'],
        styles['fed_odor_neg_taste_pos']['shape'], styles['fed_odor_pos_taste_neg']['shape']
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

def plot_centroid_vectors(latent, labels, class_names, task, ax, plot_labels=True, colors=None):
    """
    Plot vectors from origin to class-group centroids based on task-specific groupings.
    """
    if task == 'MetabolicState_2':
        group_map = {'Starved': [0],'Fed': [1]}
    elif task == 'State_Modality_6':
        group_map = {'Starved': [0, 2, 4], 'Fed': [1, 3, 5], 'Odor': [0, 1], 'Taste': [2, 3], 'Odor+Taste': [4, 5]}
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

    # if colors is None:
    #     # Define colors
    #     col_starve = 'tab:purple'
    #     col_sensory = 'tab:green'
    #     col_valence = 'tab:red'
    
    color_Starved = styles['starved']['edgecolor']
    color_Fed = styles['fed']['color']
    color_Odor = styles['fed_odor']['color']
    color_Taste = styles['fed_taste']['color']
    color_Odor_Taste = styles['fed_odor_taste']['color']
    color_Positive = '#000000'
    color_Negative = '#000000'
    color_Valence_Combi = '#000000'
    
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)


    for name, inds in group_map.items():
        mask = np.isin(labels, inds)
        if not np.any(mask):
            continue
        centroid = latent[mask].mean(axis=0)
        # choose color group
        if name == 'Starved':
            col = color_Starved
        elif name == 'Fed':
            col = color_Fed
        elif name == 'Odor':
            col = color_Odor
        elif name == 'Taste':
            col = color_Taste
        elif name == 'Odor+Taste':
            col = color_Odor_Taste
        elif name == 'Positive':
            col = color_Positive
        elif name == 'Negative':
            col = color_Negative
        elif name == 'Valence-Combi':
            col = color_Valence_Combi
        x0 = 0
        y0 = 0
        xC = centroid[0]
        yC = centroid[1]
        ax.arrow(x0, y0, dx=xC, dy=yC, head_width=4, head_length=4,color=col)
        #ax.annotate('', xy=(centroid[0], centroid[1]), xytext=(0,0),arrowprops=dict(arrowstyle='->', linewidth=1, color=col))
        offset = 14
        if plot_labels:
            ax.text(
                xC + offset*np.sign(xC), yC + offset*np.sign(yC), name,
                fontsize=8, ha='center', va='center', color=col
            )

    # Axes and formatting: show only central lines, no box or titles, fixed limits
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-70, 70)
    ax.set_ylim(-70, 70)
    ax.set_aspect('equal', 'box')
    # show only the endpoints on ticks
    ax.set_xticks([-70, 70])
    ax.set_yticks([-70, 70])


def plot_figure2(results_dict, out_path='results/CombiPlots/fig_latent_styleD.png'):
    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 3, figure=fig, left=0.08, right=0.95, wspace=0.2, hspace=0.2)
    
    # Get three tasks for three columns
    tasks = list(results_dict.keys())[:3]  # Get first three tasks
    
    # Add global row titles on the left side
    fig.text(0.02, 0.78, 'Control', rotation='vertical', va='center', fontsize=14)
    fig.text(0.02, 0.49, 'Best Run', rotation='vertical', va='center', fontsize=14)
    fig.text(0.02, 0.23, 'XXX', rotation='vertical', va='center', fontsize=14)
    
    # Plot control tSNE in first row
    for i, task in enumerate(tasks):


        run_dict = results_dict[task]
        class_names = run_dict['__class_names__']
        ctrl = 'control' if 'control' in run_dict else next(k for k in run_dict if k not in ('__class_names__','embed16_BestRun'))
        ctrl_tsne = run_dict[ctrl]['tsne_2d']
        ctrl_labels = run_dict[ctrl]['latent_labels']
        
        # Main tSNE plot
        ax = fig.add_subplot(gs[0, i])
        plot_tsne_latent(latent_2d=ctrl_tsne, labels_np=ctrl_labels,
                         class_names=class_names, ax=ax,
                         colors=TASK_COLORS[task], shapes=TASK_SHAPES[task], edgecolors=TASK_EDGECOLORS[task],
                         legend=False, draw_axis=False, draw_title=False)
        ax.set_title(f'{task}')
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False) 
        # ax.axis('off')
        
        # Create a small inset axis for the legend
        inset_ax = ax.inset_axes([0.7, 0.7, 0.4, 0.3])
        plot_centroid_vectors(ctrl_tsne, ctrl_labels, class_names, task, inset_ax, plot_labels=False)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.set_frame_on(False)

        if i != 0:
            ax.yaxis.set_visible(False)
            ax.spines['left'].set_visible(False)
    
    # Plot best run tSNE in second row
    for i, task in enumerate(tasks):
        run_dict = results_dict[task]
        class_names = run_dict['__class_names__']
        best = 'embed16_BestRun' if 'embed16_BestRun' in run_dict else ctrl
        best_tsne = run_dict[best]['tsne_2d']
        best_labels = run_dict[best]['latent_labels']
        
        # Main tSNE plot
        ax = fig.add_subplot(gs[1, i])
        plot_tsne_latent(latent_2d=best_tsne, labels_np=best_labels,
                         class_names=class_names, ax=ax, xlim=[-115, 125], ylim=[-115, 125],
                         colors=TASK_COLORS[task], shapes=TASK_SHAPES[task], edgecolors=TASK_EDGECOLORS[task],
                         legend=False, draw_axis=False, draw_title=False)
        # ax.axis('off')
        
        # Create a small inset axis for the legend [x0, y0, width, height]
        inset_ax = ax.inset_axes([0.7, 0.7, 0.4, 0.3])
        plot_centroid_vectors(best_tsne, best_labels, class_names, task, inset_ax)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.set_frame_on(False)




        if i != 0:
            ax.yaxis.set_visible(False) 
            ax.spines['left'].set_visible(False)   
    # Plot placeholder images in third row
    placeholder_path = 'src/src_imgs/Placeholder.png'
    for i, task in enumerate(tasks):
        ax = fig.add_subplot(gs[2, i])
        if os.path.exists(placeholder_path):
            img = plt.imread(placeholder_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'Placeholder', ha='center', va='center')
        ax.axis('off')
    
    # Remove tight_layout since we're using constrained_layout
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved to {out_path}")

if __name__ == '__main__':
    results = load_all_results(BASE_RESULTS_DIR)
    plot_figure2(results)
