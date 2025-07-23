import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add root directory to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.visualize_preformance import plot_tsne_latent
from src.utils.helpers import compute_tsne

# Directory containing pickle files
dir_latent_pickles = '/rhomes/aabdel/DrosoEmbedding/results/LatentSummaryPickles'
save_path = '/rhomes/aabdel/DrosoEmbedding/results/CombiPlots'

FILENAME_RE = re.compile(r'^(?P<task>[^_]+)_cnn(?P<cnn>\d+)_trf(?P<trf>\d+)_\d+\.pkl$')

# Collect results in a nested dict: {task: {cnn_dim: {trf_dim: [dicts]}}}
results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# Load all pickle files
for filename in os.listdir(dir_latent_pickles):
    if filename.endswith('.pkl') or filename.endswith('.pickle'):
        m = FILENAME_RE.match(filename)
        if not m:
            print(f"Skipping file: {filename}")
            continue

        info = m.groupdict()
        task = info['task']
        cnn_dim = int(info['cnn'])
        trf_dim = int(info['trf'])

        file_path = os.path.join(dir_latent_pickles, filename)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            # Calculate accuracy from confusion matrix
            cm = data['confusion_matrix']
            accuracy = np.sum(np.diag(cm)) / np.sum(cm)
            results[task][cnn_dim][trf_dim].append({
                'accuracy': accuracy,
                'data': data,
                'filename': filename
            })

# Get the best run for each combination
def get_best_run(runs):
    if not runs:
        return None
    return max(runs, key=lambda x: x['accuracy'])

# Create plots for each task
for task in results:
    # Get all unique CNN and TRF dimensions
    cnn_dims = sorted(set(results[task].keys()))
    trf_dims = sorted(set(trf_dim for cnn_dict in results[task].values() for trf_dim in cnn_dict.keys()))
    
    # Calculate number of rows and columns for subplots
    n_models = len(cnn_dims) * len(trf_dims)
    n_cols = 2  # CNN and Transformer plots
    n_rows = n_models  # Each model gets one row with both plots
    
    # Create figure with appropriate size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    # Ensure axes is always a flat list
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each model's latent space
    plot_idx = 0
    for cnn_dim in cnn_dims:
        for trf_dim in trf_dims:
            runs = results[task][cnn_dim][trf_dim]
            if runs:
                best_run = get_best_run(runs)
                if best_run:
                    data = best_run['data']
                    accuracy = best_run['accuracy']
                    
                    # Get latent spaces and labels
                    cnn_latent = data['cnn_latent_space']
                    transformer_latent = data['transformer_latent_space']
                    labels = data['latent_labels']
                    class_names = ["Odor (S)","Odor (F)", "Taste (S)", "Taste (F)", "Odor + Taste (S)", "Odor + Taste (F)"]
                    
                    print(f"Plotting {task} - CNN{cnn_dim} TRF{trf_dim} (Acc: {accuracy:.3f})") 
                    # Apply t-SNE to CNN latent space
                    cnn_tsne = compute_tsne(cnn_latent)
                    print(f"Computed t-SNE for CNN Latent Space")
                    
                    # Plot CNN latent space
                    plot_tsne_latent(
                        latent_2d=cnn_tsne,
                        labels_np=labels,
                        class_names=class_names,
                        ax=axes[plot_idx],
                        title=f'CNN{cnn_dim} TRF{trf_dim} (Acc: {accuracy:.3f}) - CNN Latent Space',
                        legend=False,
                        draw_axis=True,
                        draw_title=True
                    )
                    plot_idx += 1
                    
                    # Apply t-SNE to Transformer latent space
                    transformer_tsne = compute_tsne(transformer_latent)
                    print(f"Computed t-SNE for Transformer Latent Space")

                    # Plot Transformer latent space
                    plot_tsne_latent(
                        latent_2d=transformer_tsne,
                        labels_np=labels,
                        class_names=class_names,
                        ax=axes[plot_idx],
                        title=f'CNN{cnn_dim} TRF{trf_dim} (Acc: {accuracy:.3f}) - Transformer Latent Space',
                        legend=False,
                        draw_axis=True,
                        draw_title=True
                    )
                    plot_idx += 1
                    print(f"Plotted {task} - CNN{cnn_dim} TRF{trf_dim} (Acc: {accuracy:.3f})")
    
    # Add main title
    plt.suptitle(f'{task} - Latent Space Visualization', fontsize=16)
    
    # Save the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit suptitle
    plt.savefig(os.path.join(save_path, f'{task}_latent_spaces.png'))
    plt.close(fig)