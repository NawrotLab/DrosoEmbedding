import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
import random

dir_latent_pickles =  '/rhomes/aabdel/DrosoEmbedding/results/LatentSummaryPickles'
plot_out_dir = '/rhomes/aabdel/DrosoEmbedding/results/CombiPlots'

def plot_accuracy_scatter(data, task, plot_type='scatter'):
    """
    Create a scatter or contour plot with:
    x-axis: Transformer dimension
    y-axis: CNN dimension
    Color: Accuracy (with colorbar)
    
    Args:
        data: List of dictionaries containing accuracy data
        task: Name of the task
        plot_type: Either 'scatter' or 'contour'
    """
    
    # Extract all unique dimensions
    trf_dims = sorted(set(d['trf_dim'] for d in data))
    cnn_dims = sorted(set(d['cnn_dim'] for d in data))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create color map
    cmap = cm.get_cmap('viridis')
    
    if plot_type == 'scatter':
        # Store points for each dimension combination to add jitter
        points = defaultdict(list)
        
        # Collect all points
        for d in data:
            print(d['trf_dim'], d['cnn_dim'], d['accuracy'], '\n')
            points[(d['trf_dim'], d['cnn_dim'])].append(d['accuracy'])
        print("points \n", points, "\n")
        
        # Calculate min and max accuracy values for dynamic range
        all_accuracies = [acc for accs in points.values() for acc in accs]
        min_acc = min(all_accuracies) if all_accuracies else 0
        max_acc = max(all_accuracies) if all_accuracies else 100
        print(min_acc, max_acc)
        norm = plt.Normalize(min_acc, max_acc)
        
        # Create scatter plot
        sc = None
        for (trf_dim, cnn_dim), accuracies in points.items():
            # Add small random jitter to prevent overlapping points
            x_jitter = [trf_dim + random.uniform(-1, 1) for _ in accuracies]
            y_jitter = [cnn_dim + random.uniform(-1, 1) for _ in accuracies]
            
            # Plot points with color based on accuracy
            sc = ax.scatter(x_jitter, y_jitter, 
                           c=accuracies, 
                           cmap=cmap, 
                           norm=norm, 
                           alpha=0.8, 
                           s=50)
        
        # Add colorbar with dynamic range
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Accuracy')
    elif plot_type == 'contour':
        # Create grid of dimensions
        X, Y = np.meshgrid(trf_dims, cnn_dims)
        
        # Initialize Z with zeros
        cell_values = defaultdict(list) 
        # Z = np.full((len(cnn_dims), len(trf_dims)), np.nan)
        Z = np.zeros((len(cnn_dims), len(trf_dims)))


        
        # Calculate average accuracy for each dimension combination
        for d in data:
            trf_idx = trf_dims.index(d['trf_dim'])
            cnn_idx = cnn_dims.index(d['cnn_dim'])
            cell_values[(cnn_idx, trf_idx)].append(d['accuracy'])
        
        for (cnn_idx, trf_idx), accuracies in cell_values.items():
            Z[cnn_idx, trf_idx] = np.mean(accuracies) 
            print(cnn_idx, trf_idx, Z[cnn_idx, trf_idx])
        
        print(f'\n{Z}')

        
        # Calculate min and max values from Z matrix for dynamic range
        min_acc = np.nanmin(Z[Z > 0]) if np.any(Z > 0) else 0.5  # Default to 0.5 if no valid values
        max_acc = np.nanmax(Z) if len(Z) > 0 else 1.0  # Default to 1.0 if no values
        
        # Create levels that match the accuracy range
        levels = np.linspace(min_acc, max_acc, 10)
        
        # Create normalization based on dynamic range
        norm = plt.Normalize(min_acc, max_acc)
        
        # Plot contour with explicit levels and normalization
        CS = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)
        
        # Add contour lines with the same levels
        contour_lines = ax.contour(X, Y, Z, levels=levels) # , colors='k', linewidths=0.5
        # ax.clabel(contour_lines, inline=True, fontsize=8)
        
        # Add colorbar with dynamic range
        sc = CS
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Accuracy')
        cbar.set_ticks(levels[::2])  # Show every other level for better readability
        
        #Set axis limits to match scatter plot
        ax.set_xlim(min(trf_dims), max(trf_dims))
        ax.set_ylim(min(cnn_dims), max(cnn_dims))

    else:
        raise ValueError(f"Unknown plot type: {plot_type}. Must be 'scatter' or 'contour'")
    
    # Set axis labels and ticks with log2 scaling
    ax.set_xlabel('Transformer Dimension (log2 scale)')
    ax.set_ylabel('CNN Dimension (log2 scale)')
    ax.set_title(f'{task} - {plot_type.title()} Plot of Accuracy by Dimension Combination')
    ax.set_xticks(trf_dims)
    ax.set_yticks(cnn_dims)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.grid(True, axis='both', which='both', alpha=0.5)
    
    # Save plot with plot type in filename
    plt.tight_layout()
    plt.savefig(os.path.join(plot_out_dir, f"{task}_{plot_type}_v2.png"))
    """
    Create a scatter plot with:
    x-axis: Transformer dimension
    y-axis: CNN dimension
    Color: Accuracy (with colorbar)
    """
    # Extract all unique dimensions
    trf_dims = sorted(set(d['trf_dim'] for d in data))
    cnn_dims = sorted(set(d['cnn_dim'] for d in data))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create color map and normalize accuracy values (60-100 range)
    cmap = cm.get_cmap('inferno') # 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    norm = plt.Normalize(70, 95)
    
    # Store points for each dimension combination to add jitter
    points = defaultdict(list)
    
    # Collect all points
    for d in data:
        points[(d['trf_dim'], d['cnn_dim'])].append(d['accuracy'])
    
    # Plot each point with jitter
    for (trf_dim, cnn_dim), accuracies in points.items():
        # Add small random jitter to prevent overlapping points
        x_jitter = [trf_dim + random.uniform(-0.5, 0.5) for _ in accuracies]
        y_jitter = [cnn_dim + random.uniform(-0.5, 0.5) for _ in accuracies]
        
        # Plot points with color based on accuracy
        sc = ax.scatter(x_jitter, y_jitter, 
                       c=accuracies, 
                       cmap=cmap, 
                       norm=norm, 
                       alpha=0.8, 
                       s=50)
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Accuracy (%)')
    
    # Set axis labels and ticks with log2 scaling
    ax.set_xlabel('Transformer Dimension (log2 scale)')
    ax.set_ylabel('CNN Dimension (log2 scale)')
    ax.set_title(f'{task} - Accuracy by Dimension Combination')
    ax.set_xticks(trf_dims)
    ax.set_yticks(cnn_dims)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.grid(True, axis='both', which='both', alpha=0.5)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(plot_out_dir, f"{task}_Accuracy_latentDimensions_Scatter_v2.png"))
    plt.close()


# No need for these functions anymore since we're using a different visualization approach

def load_data():
    """
    Load accuracy data from pickle files.
    
    Returns:
        dict: Dictionary mapping task names to lists of dictionaries containing accuracy data
    """
    results = defaultdict(list)
    
    # Regex to parse filename: go6_cnn16_trf8_2.pkl
    FILENAME_RE = re.compile(r'^(?P<task>[^_]+)_newLR_cnn(?P<cnn>\d+)_trf(?P<trf>\d+)_\d+\.pkl$')
    
    # Load all pickle files
    for filename in os.listdir(dir_latent_pickles):
        match = FILENAME_RE.match(filename)
        if not match:
            continue
            
        filepath = os.path.join(dir_latent_pickles, filename)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        # Extract task name and dimensions
        task = match.group('task')
        cnn_dim = int(match.group('cnn'))
        trf_dim = int(match.group('trf'))
        
        # Store the data
        results[task].append({
            'accuracy': data['accuracy'],
            'cnn_dim': cnn_dim,
            'trf_dim': trf_dim
        })
    
    return dict(results)

def main(plot_type='scatter'):
    """
    Main function to plot accuracy data.
    
    Args:
        plot_type: Type of plot to create ('scatter' or 'contour')
    """
    # Load data
    results = load_data()
    
    # Create output directory if it doesn't exist
    os.makedirs(plot_out_dir, exist_ok=True)

    # ----- Main plotting -----
    for task, data in results.items():
        plot_accuracy_scatter(data, task, plot_type=plot_type)
    # Regex to parse filename: go6_cnn16_trf8_2.pkl
    # FILENAME_RE = re.compile(r'^(?P<task>[^_]+)_cnn(?P<cnn>\d+)_trf(?P<trf>\d+)_\d+\.pkl$')
    FILENAME_RE = re.compile(r'^(?P<task>[^_]+)_newLR_cnn(?P<cnn>\d+)_trf(?P<trf>\d+)_\d+\.pkl$')


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
        plot_accuracy_scatter(data, task)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot accuracy data')
    parser.add_argument('--plot-type', choices=['scatter', 'contour'], default='scatter',
                      help='Type of plot to create (default: scatter)')
    args = parser.parse_args()
    
    main(plot_type=args.plot_type)
    print('sth')
    