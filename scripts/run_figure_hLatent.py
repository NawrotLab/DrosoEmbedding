"""
Latent Space Visualization Script

This script generates a figure with t-SNE visualizations and centroid vectors
for different experimental conditions and model runs.
"""

import os
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import yaml
from matplotlib.gridspec import GridSpec
from src.visualization.visualize_preformance import plot_tsne_latent
from src.utils.helpers import load_all_results, get_style


def _setup_figure() -> Tuple[plt.Figure, GridSpec]:
    """Set up the figure and grid layout."""
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    
    # More compact figure size
    fig = plt.figure(figsize=(16, 12))
    
    # Adjust grid spec with less space between columns
    gs = GridSpec(3, 4, figure=fig, 
                     left=0.08, right=0.98,  # Use more of the figure width
                     bottom=0.15, top=0.92,  # Adjust vertical spacing
                     wspace=0.15, hspace=0.25,  # Reduce space between subplots
                     width_ratios=[1, 1, 1, 0.8])
    
    return fig, gs


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
    
    color_Starved = styles['starved']['arrow_color']
    color_Fed = styles['fed']['color']

    color_Odor = styles['odor']['color']
    color_Taste = styles['taste']['color']
    color_Odor_Taste = styles['odor_taste']['color']
    
    color_Positive = styles['positive']['color']
    color_Negative = styles['negative']['color']
    color_Valence_Combi = styles['positive_negative']['color']

    for name, inds in group_map.items():
        mask = np.isin(labels, inds)
        if not np.any(mask):
            continue
        centroid = latent[mask].mean(axis=0)
        # choose color group
        # if name == 'Starved':
        #     col = color_Starved
        # elif name == 'Fed':
        #     col = color_Fed
        # elif name == 'Odor':
        #     col = color_Odor
        # elif name == 'Taste':
        #     col = color_Taste
        # elif name == 'Odor+Taste':
        #     col = color_Odor_Taste
        # elif name == 'Positive':
        #     col = color_Positive
        # elif name == 'Negative':
        #     col = color_Negative
        # elif name == 'Valence-Combi':
        #     col = color_Valence_Combi


        if name == 'S':
            col = color_Starved
        elif name == 'F':
            col = color_Fed
        elif name == 'O':
            col = color_Odor
        elif name == 'T':
            col = color_Taste
        elif name == 'O/T':
            col = color_Odor_Taste
        elif name == '+':
            col = color_Positive
        elif name == '-':
            col = color_Negative
        elif name == '+/-':
            col = color_Valence_Combi
        
        
        x0 = 0
        y0 = 0
        xC = centroid[0]
        yC = centroid[1]
        ax.arrow(x0, y0, dx=xC, dy=yC, head_width=6, head_length=6, linewidth=2, color=col, alpha = 0.7)
        #ax.annotate('', xy=(centroid[0], centroid[1]), xytext=(0,0),arrowprops=dict(arrowstyle='->', linewidth=1, color=col))
        offset = 8
        if plot_labels:
            mid_x = x0 + xC / 2
            mid_y = y0 + yC / 2
            ax.text(
                mid_x, mid_y + offset, name,
                fontsize=14, ha='center', va='center', color=col
            )  # offset*np.sign(yC)

    # Axes and formatting: show only central lines, no box or titles, fixed limits
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-70, 70)
    ax.set_ylim(-70, 70)
    ax.set_aspect('equal', 'box')
    # # show only the endpoints on ticks
    ax.set_xticks([])
    ax.set_yticks([])


def _get_group_map(task: str, class_names: List[str]) -> Dict[str, List[int]]:
    """Return the group mapping for the given task and class names."""
    if task == 'MetabolicState_2':
        # return {'Starved': [0], 'Fed': [1]}
        return {
            'S': [0], 
            'F': [1]
        }
    
    if task == 'State_Modality_6':
        # return {
        #     'Starved': [0, 2, 4], 
        #     'Fed': [1, 3, 5], 
        #     'Odor': [0, 1], 
        #     'Taste': [2, 3], 
        #     'Odor+Taste': [4, 5]
        # }
        return {
            'S': [0, 2, 4], 
            'F': [1, 3, 5], 
            'O': [0, 1], 
            'T': [2, 3], 
            'O/T': [4, 5]
        }
    
    if task == 'State_Modality_Valence_16':
        # Build Pos/Neg based on name matching
        pos_inds = [i for i, n in enumerate(class_names) if '+' in n and '-' not in n]
        neg_inds = [i for i, n in enumerate(class_names) if '-' in n and '+' not in n]
        # return {
        #     'Starved': [0, 1, 4, 5, 8, 9, 10, 11],
        #     'Fed': [2, 3, 6, 7, 12, 13, 14, 15],
        #     'Odor': [0, 1, 2, 3],
        #     'Taste': [4, 5, 6, 7],
        #     'Odor+Taste': [8, 9, 10, 11, 12, 13],
        #     'Positive': pos_inds,
        #     'Negative': neg_inds,
        #     'Valence-Combi': [10, 11, 14, 15]
        # }
        return {
            'S': [0, 1, 4, 5, 8, 9, 10, 11],
            'F': [2, 3, 6, 7, 12, 13, 14, 15],
            'O': [0, 1, 2, 3],
            'T': [4, 5, 6, 7],
            'O/T': [8, 9, 10, 11, 12, 13],
            '+': pos_inds,
            '-': neg_inds,
            '+/-': [10, 11, 14, 15]
        }
    
    return {}


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

    # Set chance levels based on task name
    if task_name == 'State':
        chance_level = 50.0
    elif task_name == 'State, Modality':
        chance_level = 100/6  # ~16.67%
    elif task_name == 'State, Modality, Valence':  # 16 class
        chance_level = 100/16  # 6.25%
    else: 
        chance_level = 0  # 6.25%
    
    # Add value labels next to points
    for i, (mean, std, n) in enumerate(zip(means, stds, ns)):
        ax.text(i, mean - std - 2, f'{mean:.1f}±{std:.1f}\nn={n}', 
                ha='left', va='top', fontsize=7, color='k')
    # Calculate y-axis limits
    y_min = min(min(means) - 10, chance_level)  # At least show chance level
    y_max = min(max(means) + max(stds), 100)  # Cap at 100%
        
    

    
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

    ax.set_ylim(y_min, y_max)
    y_spine_min = np.floor((min(means) - 10)/10)*10
    y_spine_max = np.ceil(y_max/10)*10
    
    # Adjust spines to only show between data ranges
    ax.spines['left'].set_bounds(y_min, y_max)
    ax.spines['bottom'].set_bounds(min(x_pos), max(x_pos))
    
    # Remove top and right spines, keep left and bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set y-ticks at every 10% starting from chance level
    y_ticks = np.arange(np.ceil(chance_level/10)*10, min(101, np.floor(y_max/10)*10 + 1), 10, dtype=int)
    # Set axis limits
    
    
    
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

def _plot_tsne(
    fig: plt.Figure,
    gs: GridSpec,
    row: int,
    col: int,
    run_dict: Dict,
    class_names: List[str],
    colors: Dict,
    edges: Dict,
    shapes: Dict,
    styles: Dict = None,
    use_l_axis: bool = False,
    is_control: bool = False,
    plot_centroids: bool = False,
    task: str = None
) -> None:
    """Plot t-SNE visualization with optional centroids.
    
    Args:
        fig: Figure to plot on
        gs: GridSpec for layout
        row: Grid row
        col: Grid column
        run_dict: Data for plotting
        class_names: List of class names
        colors: Color mapping
        edges: Edge colors
        shapes: Marker shapes
        styles: Style dictionary for centroids
        use_l_axis: Use L-shaped axis
        is_control: If control plot
        plot_centroids: Add centroid vectors
        task: Task name for centroids
    """
    # Get data
    data_key = 'ControlRun' if is_control else \
              next((k for k in ['embed16_BestRun', 'control'] if k in run_dict),
                  next(k for k in run_dict if k != '__class_names__'))
    data = run_dict[data_key]
    
    # Create subplot
    ax = fig.add_subplot(gs[row, col])
    
    # Only plot t-SNE if not in centroids column (col != 2)
    if not plot_centroids:
        plot_tsne_latent(
            latent_2d=data['tsne_2d'],
            labels_np=data['latent_labels'],
            class_names=class_names,
            ax=ax,
            xlim=[-115, 125] if not use_l_axis and not is_control else None,
            ylim=[-115, 125] if not use_l_axis and not is_control else None,
            colors=colors,
            shapes=shapes,
            edgecolors=edges,
            legend=False,
            draw_axis=not use_l_axis,
            draw_title=False
        )
    
    # Add centroids if requested
    if plot_centroids and styles and task:
        plot_centroid_vectors(
            latent=data['tsne_2d'],
            labels=data['latent_labels'],
            class_names=class_names,
            task=task,
            ax=ax,
            plot_labels=False,
            styles=styles
        )
        ax.plot(0, 0, 'ko', markersize=3, zorder=10)
        ax.set_xlim(-50, 67)
        ax.set_ylim(-65, 65)
    
    # Style axes
    if use_l_axis:
        _add_l_shaped_axis(ax, show_labels=(is_control and row == 0))
    if is_control:
        ax.set_ylabel('')

def plot_figure_latent(
    results_dict: Dict,
    styles: Dict,
    colors: Dict,
    edges: Dict,
    shapes: Dict,
    out_path: str = 'results/CombiPlots/fig_latent_styleD.png',
    use_l_axis: bool = True
) -> None:
   
    """Generate the complete figure with t-SNE plots and accuracy vs dimension."""
    # Setup
    fig, gs = _setup_figure()
    
    tasks = list(results_dict.keys())[:3]
    task_names = ['State', 'State, Modality', 'State, Modality, Valence']
    task_y_pos = [0.8, 0.54, 0.26]
    
    column_titles = ['Control t-SNE', 'Model t-SNE', 'Centroids', 'Accuracy']
    column_positions = [0.2, 0.44, 0.66, 0.9] 

    for x, title in zip(column_positions, column_titles):
        fig.text(x, 0.95, title, ha='center', va='center', fontsize=12, weight='bold')
    
    # Plot each task
    for row, (task, task_name) in enumerate(zip(tasks, task_names)):
        run_dict = results_dict[task]
        class_names = run_dict.get('__class_names__', [f'Class {i}' for i in range(6)])
        
        # Add row label
        fig.text(0.05, task_y_pos[row], task_name, ha='left', va='center', fontsize=12, rotation=90, weight='bold', transform=fig.transFigure)
        
        # Plot control t-SNE (first column)
        _plot_tsne(fig, gs, row, 0, run_dict, class_names, colors[task], edges[task], shapes[task], use_l_axis=use_l_axis, is_control=True)
        
        # Plot best t-SNE (second column)
        _plot_tsne(fig, gs, row, 1, run_dict, class_names, colors[task], edges[task], shapes[task], use_l_axis=use_l_axis)
        
        # Plot centroids (third column)
        _plot_tsne(fig, gs, row, 2, run_dict, class_names, colors[task], edges[task], shapes[task], styles=styles, use_l_axis=use_l_axis, plot_centroids=True, task=task)
        
        # Plot accuracy vs dimension for each task
        ax = fig.add_subplot(gs[row, 3])
        _plot_accuracy_vs_dimension(ax, run_dict, task_name, list(colors.values())[row], row=row)
    
    # Save figure
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {out_path}")



def main():

    BASE_RESULTS_DIR = os.path.join('results', '_SeedFinals')
       
    styles, TASK_CLASS_NAMES, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES = get_style(style = "stylesE")

    results_dict = load_all_results(BASE_RESULTS_DIR, TASK_CLASS_NAMES)

    plot_figure_latent(results_dict, styles, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES, out_path='results/CombiPlots/fig_hLatent_styleE_v4.png')

if __name__ == '__main__':
    main()
