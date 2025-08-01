"""
Latent Space Visualization Script

This script generates a figure with t-SNE visualizations and centroid vectors
for different experimental conditions and model runs.
"""

import os
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import yaml
from matplotlib.gridspec import GridSpec
from src.visualization.visualize_preformance import plot_tsne_latent
from src.utils.helpers import load_all_results


def _get_group_map(task: str, class_names: List[str]) -> Dict[str, List[int]]:
    """Return the group mapping for the given task and class names."""
    if task == 'MetabolicState_2':
        return {'Starved': [0], 'Fed': [1]}
    
    if task == 'State_Modality_6':
        return {
            'Starved': [0, 2, 4], 
            'Fed': [1, 3, 5], 
            'Odor': [0, 1], 
            'Taste': [2, 3], 
            'Odor+Taste': [4, 5]
        }
    
    if task == 'State_Modality_Valence_16':
        # Build Pos/Neg based on name matching
        pos_inds = [i for i, n in enumerate(class_names) if '+' in n and '-' not in n]
        neg_inds = [i for i, n in enumerate(class_names) if '-' in n and '+' not in n]
        return {
            'Starved': [0, 1, 4, 5, 8, 9, 10, 11],
            'Fed': [2, 3, 6, 7, 12, 13, 14, 15],
            'Odor': [0, 1, 2, 3],
            'Taste': [4, 5, 6, 7],
            'Odor+Taste': [8, 9, 10, 11, 12, 13],
            'Positive': pos_inds,
            'Negative': neg_inds,
            'Valence-Combi': [10, 11, 14, 15]
        }
    
    return {}

def plot_centroid_vectors(
    latent: np.ndarray, 
    labels: np.ndarray, 
    class_names: List[str], 
    task: str, 
    ax: plt.Axes, 
    plot_labels: bool = True, 
    styles: Optional[Dict[str, Any]] = None) -> None:
    
    """
    Plot vectors from origin to class-group centroids based on task-specific groupings.
    
    Args:
        latent: 2D array of shape (n_samples, 2) containing the latent coordinates
        labels: 1D array of shape (n_samples,) containing class labels
        class_names: List of class names
        task: Name of the task (determines grouping)
        ax: Matplotlib axis to plot on
        plot_labels: Whether to plot group labels
        styles: Dictionary containing style information for different groups
    """
    styles = styles 
    group_map = _get_group_map(task, class_names)
    
    color_Starved = styles['starved']['edgecolor']
    color_Fed = styles['fed']['color']
    color_Odor = styles['fed_odor']['color']
    color_Taste = styles['fed_taste']['color']
    color_Odor_Taste = styles['fed_odor_taste']['color']
    color_Positive = '#000000'
    color_Negative = '#000000'
    color_Valence_Combi = '#000000'

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
        offset = 10
        if plot_labels:
            ax.text(
                xC + offset*np.sign(xC), yC + offset*np.sign(yC), name,
                fontsize=14, ha='center', va='center', color=col
            )

    # Axes and formatting: show only central lines, no box or titles, fixed limits
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-70, 70)
    ax.set_ylim(-70, 70)
    ax.set_aspect('equal', 'box')
    # # show only the endpoints on ticks
    ax.set_xticks([])
    ax.set_yticks([])

def _setup_figure(include_dim_plot: bool = False) -> Tuple[plt.Figure, GridSpec]:
    """Set up the figure and grid layout.
    
    Args:
        include_dim_plot: Whether to include a fourth column for dimension plots
    """
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    
    # More compact figure size
    width = 16 if include_dim_plot else 14
    fig = plt.figure(figsize=(width, 12))
    
    # Adjust grid spec with less space between columns
    if include_dim_plot:
        gs = GridSpec(3, 4, figure=fig, 
                     left=0.08, right=0.98,  # Use more of the figure width
                     bottom=0.15, top=0.92,  # Adjust vertical spacing
                     wspace=0.15, hspace=0.25,  # Reduce space between subplots
                     width_ratios=[1, 1, 1, 0.8])
    else:
        gs = GridSpec(3, 3, figure=fig,
                     left=0.08, right=0.98,
                     bottom=0.15, top=0.92,
                     wspace=0.15, hspace=0.25)
    
    return fig, gs

def _add_column_titles(fig: plt.Figure, include_dim_plot: bool = False) -> None:
    """Add column titles to the figure.
    
    Args:
        fig: Figure to add titles to
        include_dim_plot: Whether to include title for the dimension plot column
    """
    # Define column titles and their x-positions
    titles = ['Control t-SNE', 'Best Run t-SNE', 'Centroids']
    x_positions = [0.2, 0.5, 0.8]  # Normalized figure coordinates
    
    if include_dim_plot:
        titles.append('Accuracy')
        x_positions = [0.15, 0.4, 0.65, 0.9]  # Adjusted for 4 columns
    
    for x, title in zip(x_positions, titles):
        fig.text(x, 0.95, title, 
                ha='center', va='center', fontsize=12, weight='bold')
    
    # # Add x-axis label for the dimension plot
    # if include_dim_plot:
    #     fig.text(x_positions[-1], 0.05, 'Latent Dimension', 
    #             ha='center', va='center', fontsize=10)

def _plot_accuracy_vs_dimension(ax: plt.Axes, task_results: Dict, task_name: str, color: str, row: int = 0) -> None:
    """Plot accuracy vs. latent dimension as a bar plot.
    
    Args:
        ax: Matplotlib axis to plot on
        task_results: Dictionary containing task results including dim_runs
        task_name: Name of the task (for title/styling)
        color: Color to use for the bars
    """
    if 'dim_runs' not in task_results or not task_results['dim_runs']:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No dimension runs\navailable', 
                ha='center', va='center', fontsize=10)
        return
    
    # Extract accuracy data for each dimension
    dim_accuracies = {}
    for dim_key, runs in task_results['dim_runs'].items():
        dim = runs[0]['dimension']  # All runs in this list have same dimension
        # Get accuracy from each run, handling different possible result structures
        accuracies = []
        for run in runs:
            # Try different possible accuracy keys in order of preference
            if 'test_accuracy' in run['data']:
                acc = run['data']['test_accuracy']
            elif 'accuracy' in run['data']:
                acc = run['data']['accuracy']
            elif 'metrics' in run['data'] and 'test_accuracy' in run['data']['metrics']:
                acc = run['data']['metrics']['test_accuracy']
            else:
                continue  # Skip if we can't find accuracy
                
            if acc is not None:  # Only add non-None accuracies
                accuracies.append(acc)
        
        if accuracies:
            dim_accuracies[dim] = {
                'mean': np.mean(accuracies) * 100,  # Convert to percentage
                'std': np.std(accuracies, ddof=1) * 100 if len(accuracies) > 1 else 0,
                'n': len(accuracies)
            }
    
    if not dim_accuracies:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No valid accuracy data\nfound', 
                ha='center', va='center', fontsize=10)
        return
    
    # Sort dimensions and get data for plotting
    dims = sorted(dim_accuracies.keys())
    means = [dim_accuracies[d]['mean'] for d in dims]
    stds = [dim_accuracies[d]['std'] for d in dims]
    ns = [dim_accuracies[d]['n'] for d in dims]
    
    # Create line plot with error bars and markers
    x_pos = np.arange(len(dims))
    line = ax.plot(x_pos, means, 'o-', color='k', markersize=8, linewidth=2, 
                  markerfacecolor='white', markeredgecolor='k', markeredgewidth=1.5)
    
    # Add error bars
    ax.errorbar(x_pos, means, yerr=stds, fmt='none', ecolor='k', 
               capsize=5, capthick=1.5, elinewidth=1.5, alpha=0.7)
    
    # Add value labels next to points
    for i, (mean, std, n) in enumerate(zip(means, stds, ns)):
        ax.text(i, mean + 5, f'{mean:.1f}±{std:.1f}\nn={n}', 
                ha='center', va='bottom', fontsize=8, color='k')
    
    # Set chance levels based on task name
    if 'MetabolicState_2' in task_name:
        chance_level = 50.0
    elif 'State_Modality_6' in task_name:
        chance_level = 100/6  # ~16.67%
    elif 'State_Modality_Valence_16' in task_name:  # 16 class
        chance_level = 100/16  # 6.25%
    else: 
        chance_level = 0  # 6.25%
    
    # Calculate y-axis limits
    y_min = min(min(means) - 5, chance_level)  # At least show chance level
    y_max = min(max(means) + 10, 100)  # Cap at 100%
    
    # Set axis limits
    ax.set_ylim(max(0, y_min), y_max)
    
    # Customize the plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(d) for d in dims])
    if row == 2: 
        ax.set_xlabel('Latent Dimension', fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=10)
    
    # Add a horizontal line at chance level
    ax.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.text(0.02, chance_level + 1, f'Chance: {chance_level:.1f}%', 
            transform=ax.get_yaxis_transform(), color='gray', va='bottom', fontsize=8)
    
    # Adjust spines to only show between data ranges
    ax.spines['left'].set_bounds(y_min, y_max)
    ax.spines['bottom'].set_bounds(min(x_pos), max(x_pos))
    
    # Remove top and right spines, keep left and bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set y-ticks at every 10% starting from chance level
    y_ticks = np.arange(np.ceil(chance_level/10)*10, min(101, np.ceil(y_max/10)*10 + 1), 10, dtype=int)
    ax.yaxis.set_ticks(y_ticks)
    
    # Set x-ticks at each dimension value
    ax.xaxis.set_ticks(x_pos)
    ax.set_xticklabels([str(d) for d in dims])
    
    # Only show x-axis labels on the bottom plot
    if row < 2:  # For the first two rows
        ax.set_xticklabels([])  # Remove x-tick labels
        ax.spines['bottom'].set_visible(False)  # Hide bottom spine
        ax.tick_params(axis='x', which='both', length=0)  # Hide x-ticks
    else:  # For the bottom row
        ax.spines['bottom'].set_position(('outward', 10))  # Move x-axis down slightly
        ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)  # Ensure x-ticks are visible

def _add_l_shaped_axis(ax: plt.Axes, axis_length: float = 20.0, show_labels: bool = True) -> None:
    """Add L-shaped corner axis to the plot with consistent 10-unit length.
    
    Args:
        ax: Matplotlib axis to modify
        axis_length: Length of the L-shape in data coordinates (default: 10 units)
        show_labels: Whether to show the axis labels
    """
    # Get current axis limits from the plot
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    # Calculate the offset for labels (5% of the axis range)
    x_offset = (xmax - xmin) * 0.05
    y_offset = (ymax - ymin) * 0.05
    
    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add L-shaped corner axis with consistent 10-unit length
    # Horizontal line (10 units long)
    ax.axhline(y=ymin, xmin=0, xmax=axis_length/(xmax-xmin), 
               color='black', linewidth=1, clip_on=False)
    # Vertical line (10 units tall)
    ax.axvline(x=xmin, ymin=0, ymax=axis_length/(ymax-ymin), 
               color='black', linewidth=1, clip_on=False)
    
    # Add axis labels at the ends of the L if show_labels is True
    if show_labels:
        ax.text(xmin, ymin - y_offset, 't-SNE 1', 
                ha='left', va='top', fontsize=10)
        ax.text(xmin - x_offset, ymin, 't-SNE 2', 
                ha='right', va='bottom', fontsize=10, rotation=90)
    
    # Hide default ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Make sure the axis limits stay the same
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

def _plot_control_tsne(fig: plt.Figure, gs: GridSpec, row: int, 
                     run_dict: Dict, class_names: List[str], 
                     colors: Dict, edges: Dict, shapes: Dict,
                     use_l_axis: bool = False) -> plt.Axes:
    """Plot the control t-SNE visualization.
    
    Args:
        use_l_axis: If True, use L-shaped corner axis instead of standard axes
    """
    # ctrl = 'ControlRun' if 'ControlRun' in run_dict else \
    #        next(k for k in run_dict if k not in ('__class_names__', 'embed16_BestRun'))

    ctrl = 'ControlRun'
    
    ax = fig.add_subplot(gs[row, 0])
    plot_tsne_latent(
        latent_2d=run_dict[ctrl]['tsne_2d'], 
        labels_np=run_dict[ctrl]['latent_labels'],
        class_names=class_names, 
        ax=ax,
        colors=colors, 
        shapes=shapes, 
        edgecolors=edges,
        legend=False, 
        draw_axis=not use_l_axis,  # Don't draw axis if using L-shape
        draw_title=False
    )
    
    if use_l_axis:
        # Only show labels for first row, first column
        _add_l_shaped_axis(ax, show_labels=(row == 0))
    
    # No y-label for any row in the first column
    ax.set_ylabel('')
    return ax

def _plot_best_tsne(fig: plt.Figure, gs: GridSpec, row: int, 
                  run_dict: Dict, class_names: List[str], 
                  colors: Dict, edges: Dict, shapes: Dict,
                  use_l_axis: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Plot the best run t-SNE visualization and return the data.
    
    Args:
        use_l_axis: If True, use L-shaped corner axis instead of standard axes
    """
    best = 'embed16_BestRun' if 'embed16_BestRun' in run_dict else \
           'control' if 'control' in run_dict else \
           next(k for k in run_dict if k != '__class_names__')
    
    best_data = run_dict[best]
    ax = fig.add_subplot(gs[row, 1])
    plot_tsne_latent(
        latent_2d=best_data['tsne_2d'], 
        labels_np=best_data['latent_labels'],
        class_names=class_names, 
        ax=ax,
        xlim=[-115, 125] if not use_l_axis else None, 
        ylim=[-115, 125] if not use_l_axis else None,
        colors=colors, 
        shapes=shapes, 
        edgecolors=edges,
        legend=False, 
        draw_axis=not use_l_axis,  # Don't draw axis if using L-shape
        draw_title=False
    )
    
    if use_l_axis:
        _add_l_shaped_axis(ax, show_labels=False)
    
    return best_data['tsne_2d'], best_data['latent_labels']

def _plot_centroids(fig: plt.Figure, gs: GridSpec, row: int, 
                   best_tsne: np.ndarray, best_labels: np.ndarray, 
                   class_names: List[str], task: str, styles: Dict) -> None:
    """Plot the centroid vectors for the best run with L-shaped corner axis."""
    ax = fig.add_subplot(gs[row, 2])
    
    # Plot centroid vectors
    plot_centroid_vectors(
        latent=best_tsne,
        labels=best_labels,
        class_names=class_names,
        task=task,
        ax=ax,
        plot_labels=True,
        styles=styles
    )
    
    # Add a black dot at (0,0)
    ax.plot(0, 0, 'ko', markersize=3, zorder=10)
    
    # Set axis limits (assuming -70 to 70 as in the original code)
    ax.set_xlim(-65, 65)
    ax.set_ylim(-65, 65)
    
    # Add L-shaped corner axis with labels only for the first row
    _add_l_shaped_axis(ax, show_labels=False)
    
def plot_figure_latent(
    results_dict: Dict,
    styles: Dict,
    colors: Dict,
    edges: Dict,
    shapes: Dict,
    out_path: str = 'results/CombiPlots/fig_latent_styleD.png',
    use_l_axis: bool = True,
    include_dim_plot: bool = True) -> None:
    """Generate a figure with t-SNE visualizations, centroid vectors, and accuracy vs dimension plots.
    
    Args:
        results_dict: Dictionary containing all results data
        styles: Dictionary of plot styles
        colors: Dictionary of colors for each task
        edges: Dictionary of edge colors for each task
        shapes: Dictionary of marker shapes for each task
        out_path: Output path for the figure
        use_l_axis: If True, use L-shaped corner axis for all plots
        include_dim_plot: If True, include accuracy vs dimension plot as a fourth column
    """
    # Set up figure with or without fourth column
    fig, gs = _setup_figure(include_dim_plot=include_dim_plot)
    
    # Add column titles
    _add_column_titles(fig, include_dim_plot=include_dim_plot)
    
    # Get tasks (first three tasks)
    tasks = list(results_dict.keys())[:3]
    
    # Calculate row centers using relative positions within the figure
    n_rows = 3
    # Get the total height of all rows combined (sum of height ratios)
    total_ratio = sum(gs.get_height_ratios())
    # Calculate the relative position of each row's center
    row_centers = []
    current_pos = 0.0
    
    for i in range(n_rows):
        # Calculate the center of this row (0-1 from bottom to top of figure)
        row_center = 1.0 - (current_pos + (gs.get_height_ratios()[i] / (2 * total_ratio)))
        row_centers.append(row_center)
        current_pos += gs.get_height_ratios()[i] / total_ratio
    
    for row, task in enumerate(tasks):
        if row == 0:
            taskName = 'State [2 Cls]'
        elif row == 1:
            taskName = 'State, Modality [6 Cls]'
        elif row == 2:
            taskName = 'State, Modality, Valence [16 Cls]'

        run_dict = results_dict[task]
        class_names = run_dict.get('__class_names__', [f'Class {i}' for i in range(6)])
        
        # Add row label (task name) at the calculated position
        # Position text at 5% from the left edge, centered vertically in the row
        fig.text(0.05, row_centers[row], taskName, 
                ha='left', va='center', fontsize=10, rotation=90,
                transform=fig.transFigure)
        
        # Plot control t-SNE with L-shaped axis if enabled
        _plot_control_tsne(fig, gs, row, run_dict, class_names, 
                          colors[task], edges[task], shapes[task],
                          use_l_axis=use_l_axis)
        
        # Plot best t-SNE with L-shaped axis if enabled
        best_tsne, best_labels = _plot_best_tsne(
            fig, gs, row, run_dict, class_names, 
            colors[task], edges[task], shapes[task],
            use_l_axis=use_l_axis)
        
        # Plot centroids (always uses L-shaped axis)
        _plot_centroids(fig, gs, row, best_tsne, best_labels, 
                       class_names, task, styles)
        
        # Add accuracy vs dimension plot in fourth column if enabled
        if include_dim_plot:
            ax = fig.add_subplot(gs[row, 3])
            _plot_accuracy_vs_dimension(ax, run_dict, task, colors[task], row=row)
    
    # Save the figure
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {out_path}")

def get_style(style = "stylesD"):
    
    with open(f'src/visualization/{style}.yaml', "r") as f:
        styles = yaml.safe_load(f)["styles"]

    # Task-specific class names
    TASK_CLASS_NAMES = {
        'MetabolicState_2': ["Starved", "Fed"],
        'State_Modality_6': [ "Odor (S)", "Odor (F)", "Taste (S)", "Taste (F)", "Odor + Taste (S)", "Odor + Taste (F)"],
        'State_Modality_Valence_16': [
            "O$^{+}$ (S)", "O$^{-}$ (S)", "O$^{+}$ (F)", "O$^{-}$ (F)", 
            "T$^{+}$ (S)", "T$^{-}$ (S)", "T$^{+}$ (F)", "T$^{-}$ (F)", 
            "O$^{+}$+T$^{+}$ (S)", "O$^{-}$+T$^{-}$ (S)", "O$^{-}$+T$^{+}$ (S)", "O$^{+}$+T$^{-}$ (S)", 
            "O$^{+}$+T$^{+}$ (F)", "O$^{-}$+T$^{-}$ (F)", "O$^{-}$+T$^{+}$ (F)", "O$^{+}$+T$^{-}$ (F)"]
    }
    TASK_COLORS = {
        'MetabolicState_2': [styles['starved']['color'], styles['fed']['color']],
        'State_Modality_6': [
            styles['starved_odor']['color'], styles['fed_odor']['color'],
            styles['starved_taste']['color'], styles['fed_taste']['color'],
            styles['starved_odor_taste']['color'], styles['fed_odor_taste']['color']],

        'State_Modality_Valence_16': [
            styles['starved_odor_positive']['color'], styles['starved_odor_negative']['color'],
            styles['starved_taste_positive']['color'], styles['starved_taste_negative']['color'],
            styles['starved_odor_pos_taste_pos']['color'], styles['starved_odor_neg_taste_neg']['color'],
            styles['starved_odor_neg_taste_pos']['color'], styles['starved_odor_pos_taste_neg']['color'],
            styles['fed_odor_positive']['color'], styles['fed_odor_negative']['color'],
            styles['fed_taste_positive']['color'], styles['fed_taste_negative']['color'],
            styles['fed_odor_pos_taste_pos']['color'], styles['fed_odor_neg_taste_neg']['color'],
            styles['fed_odor_neg_taste_pos']['color'], styles['fed_odor_pos_taste_neg']['color']]
    }

    TASK_EDGECOLORS = {
        'MetabolicState_2': [styles['starved']['edgecolor'], styles['fed']['edgecolor']],
        'State_Modality_6': [
            styles['starved_odor']['edgecolor'], styles['fed_odor']['edgecolor'],
            styles['starved_taste']['edgecolor'], styles['fed_taste']['edgecolor'],
            styles['starved_odor_taste']['edgecolor'], styles['fed_odor_taste']['edgecolor']],

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

    return styles, TASK_CLASS_NAMES, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES


def main():

    # Base results directory
    BASE_RESULTS_DIR = os.path.join('results', '_SeedFinals')
    # Columns to plot
    ROWS = ['tsne_ctrl', 'tsne_best', 'embedding_vectors', 'embedding_dimension']
    
    styles, TASK_CLASS_NAMES, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES = get_style()

    results_dict = load_all_results(BASE_RESULTS_DIR, TASK_CLASS_NAMES)

    plot_figure_latent(results_dict, styles, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES,out_path='results/CombiPlots/fig_hLatent_styleD_v3.png')

if __name__ == '__main__':
    main()
