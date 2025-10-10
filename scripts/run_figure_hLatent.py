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
from src.utils.helpers import load_all_results, get_style, _color_for_group


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

def _get_group_map(task: str, class_names: List[str]) -> Dict[str, List[int]]:
    """Return the group mapping for the given task and class names."""
    if task == 'MetabolicState_2':
        return {
            'S': [0], 
            'F': [1]
        }
    
    if task == 'State_Modality_6':
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



def plot_centroid_vectors(
    latent: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    task: str,
    ax: plt.Axes,
    plot_labels: bool = True,
    styles: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Plot displacement vectors from the global mean to task-defined group centroids
    in a 2D latent space (e.g., t-SNE). Re-centers so the global mean is at (0, 0).
    latent: (n_samples, 2); labels: (n_samples,)
    """
    assert latent.ndim == 2 and latent.shape[1] == 2, "latent must be (N, 2)"
    assert labels.shape[0] == latent.shape[0], "labels length must match latent"

    # 1) Center so that global mean is the origin
    global_mu = latent.mean(axis=0)
    Z = latent - global_mu  # (N, 2)

    # 2) Task-specific grouping: map short codes -> indices
    group_map = _get_group_map(task, class_names)  # e.g., {'S':[...], 'F':[...], 'O':[...] ...}

    # 3) Colors: unify lookup with fallback to avoid KeyError
    def _col(key: str) -> Any:
        if styles is None:
            return "k"
        d = styles.get(key, {})
        return d.get("arrow_color", d.get("color", "k"))

    # Optional: a stable draw order (longer vectors underneath shorter so labels sit on top)
    # Compute centroids first
    centroids = {}
    for name, inds in group_map.items():
        mask = np.isin(labels, inds)
        if np.any(mask):
            centroids[name] = Z[mask].mean(axis=0)  # mu_g - bar_mu

    # 4) Plot arrows
    for name, vec in centroids.items():
        col = (
            _col("starved")            if name == "S"   else
            _col("fed")                if name == "F"   else
            _col("odor")               if name == "O"   else
            _col("taste")              if name == "T"   else
            _col("odor_taste")         if name == "O/T" else
            _col("positive")           if name == "+"   else
            _col("negative")           if name == "-"   else
            _col("positive_negative")  if name == "+/-" else
            "k"
        )
        x0, y0 = 0.0, 0.0
        dx, dy = float(vec[0]), float(vec[1])
        ax.arrow(
            x0, y0, dx, dy,
            head_width=6, head_length=6, linewidth=2,
            alpha=0.7, color=col, length_includes_head=True, zorder=2
        )
        if plot_labels:
            # Label at mid-point with a fixed 8pt vertical offset (screen space)
            mid_x, mid_y = x0 + dx / 2.0, y0 + dy / 2.0
            ax.annotate(
                name, xy=(mid_x, mid_y), xytext=(0, 8),
                textcoords="offset points", ha="center", va="center",
                fontsize=14, color=col, zorder=3
            )

    # 5) Cosmetics
    # crosshair at origin (faint)
    ax.axhline(0, lw=0.8, alpha=0.15, color="0.3", zorder=1)
    ax.axvline(0, lw=0.8, alpha=0.15, color="0.3", zorder=1)

    # auto limits based on vector norms (pad 10%)
    if centroids:
        R = np.max([np.hypot(*v) for v in centroids.values()])
        R = 1.1 * R if R > 0 else 1.0
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
    ax.set_aspect("equal", adjustable="box")

    # clean frame
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_accuracy_vs_dimension(
    ax: plt.Axes,
    task_results: Dict,
    task_name: str,
    color: str,
    row: int = 0,
    primary_family: str = "E",           # "E" or "H"
    overlay_alt_family: bool = True,
    grid_dims: Optional[List[int]] = None,  # master grid for x-axis (e.g., [4,8,16,32,64])
) -> None:
    # ------------- helpers -------------
    def _extract_family_stats(family_prefix: str):
        fam_items = [(k, v) for k, v in task_results.get('runs', {}).items() if k.startswith(family_prefix)]
        if not fam_items:
            return [], {}
        dim_accuracies = {}
        for dim_key, runs in fam_items:
            try:
                dim = int(dim_key[1:])  # "E16" -> 16
            except Exception:
                continue
            accs = []
            for run in runs:
                d = run.get('data', {}) or {}
                acc = (
                    d.get('test_accuracy', None)
                    if d.get('test_accuracy', None) is not None else
                    d.get('accuracy', None)
                    if d.get('accuracy', None) is not None else
                    (d.get('metrics', {}) or {}).get('test_accuracy', None)
                )
                if acc is not None:
                    accs.append(float(acc))
            if accs:
                accs = np.asarray(accs, dtype=float)
                dim_accuracies[dim] = {
                    'best_accuracy': float(np.max(accs) * 100.0),
                    'mean': float(np.mean(accs) * 100.0),
                    'median': float(np.median(accs) * 100.0),
                    'quantiles': np.percentile(accs, [25, 75]) * 100.0,
                    'std': float(np.std(accs, ddof=1) * 100.0) if len(accs) > 1 else 0.0,
                    'n': int(len(accs)),
                }
        dims_sorted = sorted(dim_accuracies.keys())
        return dims_sorted, dim_accuracies

    # default grid if not provided
    if grid_dims is None:
        grid_dims = [2, 4, 8, 16, 32, 64]
    ix = {d: i for i, d in enumerate(grid_dims)}  # dim -> x position on the fixed grid

    # --- primary family ---
    dims_p, stats_p = _extract_family_stats(primary_family)
    if not dims_p:
        ax.axis('off'); ax.text(0.5, 0.5, f'No {primary_family}* runs', ha='center', va='center', fontsize=10)
        return

    # optional alt
    alt_family = "H" if primary_family == "E" else "E"
    dims_a, stats_a = _extract_family_stats(alt_family) if overlay_alt_family else ([], {})

    # map dims to fixed positions (skip dims not in grid)
    dims_p = [d for d in dims_p if d in ix]
    x_p = np.array([ix[d] for d in dims_p], dtype=int)
    med_p = [stats_p[d]['median'] for d in dims_p]
    q_p = [stats_p[d]['quantiles'] for d in dims_p]
    best_p = [stats_p[d]['best_accuracy'] for d in dims_p]
    n_p = [stats_p[d]['n'] for d in dims_p]
    qlow_p = np.array([q[0] for q in q_p])
    qupp_p = np.array([q[1] for q in q_p])

    # best marker by best run
    best_idx = int(np.argmax(best_p))
    best_x = x_p[best_idx]
    best_y = best_p[best_idx]

    if dims_a:
        dims_a = [d for d in dims_a if d in ix]
        if dims_a:
            x_a   = np.array([ix[d] for d in dims_a], dtype=int)
            med_a = [stats_a[d]['median']     for d in dims_a]
            q_a   = [stats_a[d]['quantiles']  for d in dims_a]   # <— add this
            qlow_a = np.array([q[0] for q in q_a])               # <— add this
            qupp_a = np.array([q[1] for q in q_a])               # <— add this

            ax.plot(
                x_a, med_a, 'o-', color='darkcyan',markersize=8, linewidth=1.5, alpha = 0.8,
                markerfacecolor='white', markeredgecolor='darkcyan', markeredgewidth=1.5,
                label=f"{alt_family}", zorder=2
            )
            ax.fill_between(x_a, qlow_a, qupp_a, alpha=0.3, color='darkcyan', linewidth=0, zorder=1)

    # draw primary
    ax.plot(x_p, med_p, 'o-', color='k', markersize=8, linewidth=1.5, alpha = 0.8,
            markerfacecolor='white', markeredgecolor='k', markeredgewidth=1.5,label=f"{primary_family}", zorder=2)
    ax.fill_between(x_p, qlow_p, qupp_p, alpha=0.5, color='gray', linewidth=0, zorder=1)
    ax.plot(best_x, best_y, marker='*', markersize=10, color='gold',
            markeredgecolor='k', markeredgewidth=0.8, zorder=5)

    # chance + cosmetics (unchanged except x-axis setup)
    if task_name == 'i. State':
        chance_level, y_min, y_max = 50.0, 70, 100
    elif task_name == 'ii. State, Modality':
        chance_level, y_min, y_max = 100/6, 70, 100
    elif task_name == 'iii. State, Modality, Valence':
        chance_level, y_min, y_max = 100/16, 60, 90
    else:
        chance_level, y_min, y_max = 0.0, 60, 100

    for x, (y, n) in zip(x_p, zip(med_p, n_p)):
        if y_min <= y <= y_max:
            ax.text(x, y - 2, f'{y:.1f}\nn={n}', ha='left', va='top', fontsize=7, color='k')

    # x-axis: show the full grid so missing dims appear as gaps
    N = len(grid_dims)
    ax.set_xticks(np.arange(N))
    ax.set_xticklabels([str(d) for d in grid_dims])
    if row == 2:
        ax.set_xlabel(f'Latent Dimension ({primary_family}*)', fontsize=10)

    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.text(0.7, y_min + 1, f'Chance: {chance_level:.1f}%',
            transform=ax.get_yaxis_transform(), color='gray', va='bottom', fontsize=8)

    # frame/limits
    ax.spines['left'].set_bounds(y_min, y_max)
    ax.spines['bottom'].set_bounds(0, max(0, N-1))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    y_ticks = np.arange(np.ceil(chance_level/10)*10, min(101, np.floor(y_max/10)*10 + 1), 10, dtype=int)
    ax.yaxis.set_ticks(y_ticks)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.5, N - 0.5)

    if row < 2:
        ax.set_xticklabels([]); ax.spines['bottom'].set_visible(False); ax.tick_params(axis='x', which='both', length=0)
    else:
        ax.spines['bottom'].set_position(('outward', 10)); ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)

    # optional legend if overlay present
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles, labels, loc="upper right", fontsize=8, frameon=False)


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
    data_key = 'control' if is_control else \
              next((k for k in ['best', 'control'] if k in run_dict),
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
    # Add centroids if requested — compute from the actual 2-D cloud
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
        ax.plot(0, 0, 'ko', markersize=3, zorder=10)  # show global mean


    # (keep the styling below exactly as you had)

    
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
    out_path: str = 'results/CombiPlots/fig_latent_chptRuns_1.png',
    use_l_axis: bool = True
) -> None:
   
    """Generate the complete figure with t-SNE plots and accuracy vs dimension."""
    # Setup
    fig, gs = _setup_figure()
    
    tasks = list(results_dict.keys())[:3]
    task_names = ['i. State', 'ii. State, Modality', 'iii. State, Modality, Valence']
    task_y_pos = [0.81, 0.53, 0.25]
    
    column_titles = ['(a) Control t-SNE', '(b) Model t-SNE', '(c) Centroids', '(d) Accuracy']
    column_positions = [0.18, 0.43, 0.65, 0.9] 

    for x, title in zip(column_positions, column_titles):
        fig.text(x, 0.95, title, ha='center', va='center', fontsize=12, weight='bold')
    
    # Plot each task
    for row, (task, task_name) in enumerate(zip(tasks, task_names)):
        run_dict = results_dict[task]
        class_names = run_dict.get('__class_names__')
        
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
        # _plot_accuracy_vs_dimension(ax, run_dict, task_name, list(colors.values())[row], row=row)
        _plot_accuracy_vs_dimension(ax, run_dict, task_name,color=list(colors.values())[row], row=row, primary_family="H", overlay_alt_family=True)
    
    # Save figure
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {out_path}")



def main():

    # BASE_RESULTS_DIR = os.path.join('results', '_archive', '_SeedFinals')
    BASE_RESULTS_DIR = os.path.join('results', '_chkpt_finals')       
    styles, TASK_CLASS_NAMES, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES = get_style(style = "stylesE")

    results_dict = load_all_results(BASE_RESULTS_DIR, TASK_CLASS_NAMES)

    plot_figure_latent(results_dict, styles, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES, out_path='results/CombiPlots/fig_Latent_chkpt_per40.png')

if __name__ == '__main__':
    main()
