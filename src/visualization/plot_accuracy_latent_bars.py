import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

dir_latent_pickles =  '/rhomes/aabdel/DrosoEmbedding/results/LatentSummaryPickles'
plot_out_dir = '/rhomes/aabdel/DrosoEmbedding/results/CombiPlots'

def plot_line_with_samples(x_vals, y_dict, title, xlabel, ax=None, row_idx=0, task='', save_path=''):
    """
    x_vals: sorted list of embedding dims (int)
    y_dict: dict of embedding_dim -> list of accuracies
    ax: matplotlib axis object (if None, creates a new figure)
    row_idx: index of the row in the subplot (unused, kept for backward compatibility)
    """
    # Calculate means and check if we have multiple samples
    means = []
    stds = []
    for x in x_vals:
        samples = y_dict[x]
        means.append(np.mean(samples))
        stds.append(np.std(samples, ddof=1) if len(samples) > 1 else 0)
    
    # Create a new figure if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot all individual points for each dim
    for x in x_vals:
        y = y_dict[x]
        # Use a different marker style if there's only one sample
        marker = 'o' if len(y) > 1 else 'x'
        ax.scatter([x]*len(y), y, s=30, color='k', alpha=0.8, zorder=3, 
                  marker=marker, label='_nolegend_')
    
    # Plot mean line with error bars if we have multiple samples
    if all(len(y_dict[x]) > 1 for x in x_vals):
        ax.errorbar(x_vals, means, yerr=stds, fmt='-o', label='Mean ± std')
    else:
        # If any x_val has only one sample, just plot the mean line
        ax.plot(x_vals, means, 'o-', label='Accuracy')
    
    ax.set_xticks([4, 8, 16, 32, 64])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.grid(True, axis='both')
    ax.legend()
    
    # Save the figure if save_path is provided and we created the figure
    if save_path and ax is None:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def plot_case_a(task, data, ax, row_idx):
    subset = [d for d in data if d['cnn_dim'] == 16]
    if not subset:
        print(f"No data for case (a) for {task}")
        return
    accs = defaultdict(list)
    for d in subset:
        accs[d['trf_dim']].append(d['accuracy'])
    x = sorted(accs.keys())
    plot_line_with_samples(x, accs, f"{task}: CNN=16, varying Transformer dim", 'Transformer Embedding Dim', ax=ax, row_idx=row_idx, task=task)

def plot_case_b(task, data, ax, row_idx):
    subset = [d for d in data if d['trf_dim'] == 16]
    if not subset:
        print(f"No data for case (b) for {task}")
        return
    accs = defaultdict(list)
    for d in subset:
        accs[d['cnn_dim']].append(d['accuracy'])
    x = sorted(accs.keys())
    plot_line_with_samples(x, accs, f"{task}: Transformer=16, varying CNN dim", 'CNN Embedding Dim', ax=ax, row_idx=row_idx, task=task)

def plot_case_c(task, data, ax, row_idx):
    subset = [d for d in data if d['cnn_dim'] == d['trf_dim']]
    if not subset:
        print(f"No data for case (c) for {task}")
        return
    accs = defaultdict(list)
    for d in subset:
        accs[d['cnn_dim']].append(d['accuracy'])
    x = sorted(accs.keys())
    plot_line_with_samples(x, accs, f"{task}: CNN=Transformer, varying both", 'Embedding Dim (CNN=Transformer)', ax=ax, row_idx=row_idx, task=task)

def main():
    # Regex to parse filename: go6_cnn16_trf8_2.pkl
    # FILENAME_RE = re.compile(r'^(?P<task>[^_]+)_cnn(?P<cnn>\d+)_trf(?P<trf>\d+)_\d+\.pkl$')
    FILENAME_RE = re.compile(r'^(?P<task>[^_]+)_cnn(?P<cnn>\d+)_trf(?P<trf>\d+)_\d+\.pkl$')


    # Collect results in a nested dict: {task: [dicts]}
    results = defaultdict(list)

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
                cm = data['confusion_matrix']
                total_correct = np.trace(cm)
                total_samples = np.sum(cm)
                accuracy = (total_correct / total_samples) * 100

            # Store all info
            results[task].append({
                'cnn_dim': cnn_dim,
                'trf_dim': trf_dim,
                'accuracy': accuracy,
                'filename': filename,
            })



    # ----- Main plotting -----
    for task, data in results.items():
        # Create directory for this task if it doesn't exist
        task_dir = os.path.join(plot_out_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        
        # Plot each case in a separate figure
        plot_case_a(task, data, None, 0)
        plt.tight_layout()
        plt.savefig(os.path.join(task_dir, f"{task}_bs256_lr0005.png"))
        plt.close()
        
        # Uncomment these if you want to plot the other cases
        # plot_case_b(task, data, None, 0)
        # plt.tight_layout()
        # plt.savefig(os.path.join(task_dir, f"{task}_case_b.png"))
        # plt.close()
        
        # plot_case_c(task, data, None, 0)
        # plt.tight_layout()
        # plt.savefig(os.path.join(task_dir, f"{task}_case_c.png"))
        # plt.close()


if __name__ == "__main__":
    main()
    print('sth')
    