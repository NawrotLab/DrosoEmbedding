import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def moving_average(data, window_size=50):
    """Calculate moving average with specified window size"""
    if len(data) < window_size:
        return data
    
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

# Directory containing pickle files
dir_latent_pickles = '/rhomes/aabdel/DrosoEmbedding/results/LatentSummaryPickles'
save_path = '/rhomes/aabdel/DrosoEmbedding/results/CombiPlots'

# FILENAME_RE = re.compile(r'^(?P<task>[^_]+)_cnn(?P<cnn>\d+)_trf(?P<trf>\d+)_\d+\.pkl$')
FILENAME_RE = re.compile(r'^(?P<task>[^_]+)_newLR_cnn(?P<cnn>\d+)_trf(?P<trf>\d+)_\d+\.pkl$')

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
    # Create a single figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all dimension combinations
    for cnn_dim in sorted(results[task]):
        for trf_dim in sorted(results[task][cnn_dim]):
            runs = results[task][cnn_dim][trf_dim]
            if runs:
                best_run = get_best_run(runs)
                if best_run:
                    data = best_run['data']
                    accuracy = best_run['accuracy']
                    
                    # Smooth the loss curve
                    loss = data['val_loss']
                    smoothed_loss = moving_average(loss)
                    
                    # Plot the loss curves
                    epochs = range(1, len(smoothed_loss) + 1)
                    ax.plot(epochs, smoothed_loss, 
                         label=f'CNN{cnn_dim} TRF{trf_dim} (Acc: {accuracy:.3f})',
                         alpha=0.8)  # Add slight transparency for better visibility
    
    # Add labels and legend
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add main title
    ax.set_title(f'{task} - Best Runs Comparison')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{task}_latent_loss_newLR.png'))
    plt.close(fig)
    print(f"Saved plot for {task}_newLR")
