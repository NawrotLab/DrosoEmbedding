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
from mpl_toolkits.mplot3d import Axes3D
from src.visualization.visualize_preformance import plot_tsne_latent
from src.utils.helpers import load_all_results, get_style, _color_for_group, scatter_bicolor, HALF_CIRCLE_LEFT, HALF_CIRCLE_RIGHT
from src.utils.logger import setup_logger


# ═══════════════════════════════════════════════
# BIOLOGICAL AXES — replaces t-SNE centroid arrows in column C
# ═══════════════════════════════════════════════

def _short_name(n: str) -> str:
    """Shorten LaTeX class names for plot labels."""
    return n.replace('$^{+}$', '+').replace('$^{-}$', '-').replace(' (S)', ' S').replace(' (F)', ' F')


def _compute_biological_axes(X, labels, task_key, class_names):
    """
    Compute biological axes and project class centroids.
    Returns: (projections_dict, axes_dict, displacements)
    """
    global_mu = X.mean(axis=0)
    unique_labels = np.sort(np.unique(labels))
    centroids = np.array([X[labels == l].mean(axis=0) for l in unique_labels])
    displacements = centroids - global_mu

    if task_key == 'MetabolicState_2':
        mean_S = X[labels == 0].mean(axis=0)
        mean_F = X[labels == 1].mean(axis=0)
        state_axis = mean_F - mean_S
        state_axis /= np.linalg.norm(state_axis)
        proj_state = displacements @ state_axis
        return {'state': proj_state}, {'state': state_axis}, displacements

    if task_key == 'State_Modality_6':
        mean_S = X[np.isin(labels, [0, 2, 4])].mean(axis=0)
        mean_F = X[np.isin(labels, [1, 3, 5])].mean(axis=0)
        mean_O = X[np.isin(labels, [0, 1])].mean(axis=0)
        mean_T = X[np.isin(labels, [2, 3])].mean(axis=0)

        state_axis = mean_F - mean_S
        state_axis /= np.linalg.norm(state_axis)
        modality_axis = mean_T - mean_O
        modality_axis /= np.linalg.norm(modality_axis)

        proj_state = displacements @ state_axis
        proj_modality = displacements @ modality_axis

        return {'state': proj_state, 'modality': proj_modality}, \
               {'state': state_axis, 'modality': modality_axis}, displacements

    if task_key == 'State_Modality_Valence_16':
        mean_S = X[np.isin(labels, [0, 1, 4, 5, 8, 9, 10, 11])].mean(axis=0)
        mean_F = X[np.isin(labels, [2, 3, 6, 7, 12, 13, 14, 15])].mean(axis=0)
        mean_O = X[np.isin(labels, [0, 1, 2, 3])].mean(axis=0)
        mean_T = X[np.isin(labels, [4, 5, 6, 7])].mean(axis=0)

        pos_inds = [i for i, n in enumerate(class_names) if '$^{+}$' in n and '$^{-}$' not in n]
        neg_inds = [i for i, n in enumerate(class_names) if '$^{-}$' in n and '$^{+}$' not in n]
        mean_pos = X[np.isin(labels, pos_inds)].mean(axis=0)
        mean_neg = X[np.isin(labels, neg_inds)].mean(axis=0)

        state_axis = mean_F - mean_S
        state_axis /= np.linalg.norm(state_axis)
        modality_axis = mean_T - mean_O
        modality_axis /= np.linalg.norm(modality_axis)
        valence_axis = mean_pos - mean_neg
        valence_axis /= np.linalg.norm(valence_axis)

        proj_state = displacements @ state_axis
        proj_modality = displacements @ modality_axis
        proj_valence = displacements @ valence_axis

        return {'state': proj_state, 'modality': proj_modality, 'valence': proj_valence}, \
               {'state': state_axis, 'modality': modality_axis, 'valence': valence_axis}, displacements


def _clean_axis(ax):
    """Remove all spines, ticks, and labels — blank canvas."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def _add_axis_indicator(ax, axis_names, fontsize=8):
    """
    Add a clean L-shaped axis indicator in the bottom-left corner.
    All lines meet at a single origin point. Uses axes-fraction transform.
    
    axis_names: list of axis name strings
      - 1 name  → horizontal line only (e.g. ['State'])
      - 2 names → L-shape: horizontal + vertical (e.g. ['State', 'Modality'])
      - 3 names → L-shape + diagonal for 3rd axis
    """
    from matplotlib.transforms import blended_transform_factory

    # Origin and arm length in axes fraction
    x0, y0 = 0.06, 0.06
    length = 0.13
    trans = ax.transAxes

    # Horizontal arm
    if len(axis_names) >= 1:
        ax.plot([x0, x0 + length], [y0, y0], '-', color='black', lw=1.0,
                transform=trans, clip_on=False)
        ax.text(x0 + length / 2, y0 - 0.035, axis_names[0],
                transform=trans, ha='center', va='top', fontsize=fontsize)

    # Vertical arm (connected at same origin)
    if len(axis_names) >= 2:
        ax.plot([x0, x0], [y0, y0 + length], '-', color='black', lw=1.0,
                transform=trans, clip_on=False)
        ax.text(x0 - 0.02, y0 + length / 2, axis_names[1],
                transform=trans, ha='right', va='center', fontsize=fontsize,
                rotation=90)

    # Diagonal arm (connected at same origin)
    if len(axis_names) >= 3:
        diag = length * 0.75
        dx = diag * np.cos(np.deg2rad(45))
        dy = diag * np.sin(np.deg2rad(45))
        ax.plot([x0, x0 + dx], [y0, y0 + dy], '-', color='black', lw=1.0,
                transform=trans, clip_on=False)
        ax.text(x0 + dx + 0.015, y0 + dy + 0.01, axis_names[2],
                transform=trans, ha='left', va='bottom', fontsize=fontsize)


def _add_cos_box(ax, lines, fontsize=10):
    """Add cosine similarity + variance annotation box in top-left corner."""
    text = '\n'.join(lines)
    ax.text(0.02, 0.98, text,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=fontsize, fontstyle='italic', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.85),
            zorder=10)


def plot_biological_axes_panel(
    ax,
    X: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    task: str,
    colors: Dict,
    edges: Dict,
    bicolor_info: Dict = None,
    fig: plt.Figure = None,
    gs: 'GridSpec' = None,
    row: int = 0,
) -> None:
    """
    Plot biological axes projections with cosine similarity annotations.
    Clean style: plain gray lines, text endpoint labels, no axis frames,
    no class name annotations, cosine + variance box in top-left.
    
    All three rows use 2D axes (16-class uses oblique projection).
    """
    if bicolor_info is None:
        bicolor_info = {}
    projections, axes_dict, displacements = _compute_biological_axes(X, labels, task, class_names)
    n_classes = len(np.unique(labels))

    # Variance explained
    total_var = np.sum(displacements ** 2)
    explained = sum(np.sum(p ** 2) for p in projections.values())
    pct = explained / total_var * 100

    # ── 2-class: 1D number line ──
    if task == 'MetabolicState_2':
        proj_s = projections['state']

        # Cosine similarity between the 2 displacement vectors
        cos_val = np.dot(displacements[0], displacements[1]) / \
                  (np.linalg.norm(displacements[0]) * np.linalg.norm(displacements[1]))

        _clean_axis(ax)

        # Gray line
        xlim = np.abs(proj_s).max() * 1.4
        ax.plot([-xlim, xlim], [0, 0], '-', color='gray', lw=1.2, alpha=0.6, zorder=1)

        # Origin cross
        ax.plot(0, 0, '+', color='gray', ms=8, mew=1.5, zorder=2)

        # Centroids (no labels)
        for i in range(n_classes):
            is_starved = '(S)' in class_names[i] or class_names[i] == 'Starved'
            lw = 2.0 if is_starved else 0.5
            if i in bicolor_info:
                scatter_bicolor(ax, proj_s[i], 0, bicolor_info[i], s=200, linewidth=lw, zorder=3)
            else:
                ax.scatter(proj_s[i], 0, c=colors[i], edgecolors=edges[i],
                           linewidth=lw, s=200, zorder=3)

        # Endpoint text labels — placed beyond the line ends for clearance
        ax.text(-xlim * 1.15, 0, 'Starved', ha='right', va='center', fontsize=10)
        ax.text(xlim * 1.15, 0, 'Fed', ha='left', va='center', fontsize=10)

        ax.set_xlim(-xlim * 2.2, xlim * 2.2)
        ax.set_ylim(-0.5, 0.5)

        # Axis indicator
        _add_axis_indicator(ax, ['State'])

    # ── 6-class: 2D scatter ──
    elif task == 'State_Modality_6':
        proj_s, proj_m = projections['state'], projections['modality']

        # Cosine similarity between axes
        cos_sm = np.dot(axes_dict['state'], axes_dict['modality'])

        _clean_axis(ax)

        # Axis limits
        all_vals = np.concatenate([proj_s, proj_m])
        lim = np.abs(all_vals).max() * 1.4

        # Plain gray crossing lines
        ax.plot([-lim, lim], [0, 0], '-', color='gray', lw=1.0, alpha=0.5, zorder=1)
        ax.plot([0, 0], [-lim, lim], '-', color='gray', lw=1.0, alpha=0.5, zorder=1)

        # Origin cross
        ax.plot(0, 0, '+', color='gray', ms=8, mew=1.5, zorder=2)

        # Centroids (no labels)
        for i in range(n_classes):
            is_starved = '(S)' in class_names[i]
            lw = 2.0 if is_starved else 0.5
            if i in bicolor_info:
                scatter_bicolor(ax, proj_s[i], proj_m[i], bicolor_info[i], s=180, linewidth=lw, zorder=3)
            else:
                ax.scatter(proj_s[i], proj_m[i], c=colors[i], edgecolors=edges[i],
                           linewidth=lw, s=180, zorder=3)

        # Endpoint text labels
        ax.text(-lim, 0, 'Starved  ', ha='right', va='center', fontsize=10)
        ax.text(lim, 0, '  Fed', ha='left', va='center', fontsize=10)
        ax.text(0, lim, 'Taste', ha='center', va='bottom', fontsize=10)
        ax.text(0, -lim, 'Odor', ha='center', va='top', fontsize=10)

        ax.set_xlim(-lim * 1.7, lim * 1.7)
        ax.set_ylim(-lim * 1.5, lim * 1.5)

        # Axis indicator
        _add_axis_indicator(ax, ['State', 'Modality'])

    # ── 16-class: flat 2D with oblique 3rd axis ──
    elif task == 'State_Modality_Valence_16':
        proj_s = projections['state']
        proj_m = projections['modality']
        proj_v = projections['valence']

        # Pairwise cosine similarities
        cos_sm = np.dot(axes_dict['state'], axes_dict['modality'])
        cos_sv = np.dot(axes_dict['state'], axes_dict['valence'])
        cos_mv = np.dot(axes_dict['modality'], axes_dict['valence'])

        _clean_axis(ax)

        # Oblique projection: state → x, modality → y, valence → diagonal
        # The diagonal angle gives the 3D feel in 2D
        angle = np.deg2rad(35)  # oblique angle for valence axis
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Project to 2D: x = state + valence*cos(angle), y = modality + valence*sin(angle)
        # Scale valence contribution so it doesn't dominate
        v_scale = 0.6
        x_pts = proj_s + proj_v * v_scale * cos_a
        y_pts = proj_m + proj_v * v_scale * sin_a

        # Axis limits
        all_vals = np.concatenate([x_pts, y_pts])
        lim = np.abs(all_vals).max() * 1.3

        # Draw three gray axis lines through origin
        # State axis: horizontal
        ax.plot([-lim, lim], [0, 0], '-', color='gray', lw=1.0, alpha=0.5, zorder=1)
        # Modality axis: vertical
        ax.plot([0, 0], [-lim, lim], '-', color='gray', lw=1.0, alpha=0.5, zorder=1)
        # Valence axis: diagonal
        diag_len = lim * 0.85
        ax.plot([-diag_len * cos_a, diag_len * cos_a],
                [-diag_len * sin_a, diag_len * sin_a],
                '-', color='gray', lw=1.0, alpha=0.5, zorder=1)

        # Origin cross
        ax.plot(0, 0, '+', color='gray', ms=8, mew=1.5, zorder=2)

        # Centroids (no labels)
        for i in range(n_classes):
            is_starved = '(S)' in class_names[i]
            lw = 2.0 if is_starved else 0.5
            if i in bicolor_info:
                scatter_bicolor(ax, x_pts[i], y_pts[i], bicolor_info[i], s=120, linewidth=lw, zorder=3)
            else:
                ax.scatter(x_pts[i], y_pts[i], c=colors[i], edgecolors=edges[i],
                           linewidth=lw, s=120, zorder=3)

        # Endpoint text labels
        ax.text(-lim, 0, 'Starved  ', ha='right', va='center', fontsize=10)
        ax.text(lim, 0, '  Fed', ha='left', va='center', fontsize=10)
        ax.text(0, lim, 'Taste', ha='center', va='bottom', fontsize=10)
        ax.text(0, -lim, 'Odor', ha='center', va='top', fontsize=10)
        ax.text(diag_len * cos_a, diag_len * sin_a, '  App.', ha='left', va='bottom', fontsize=10)
        ax.text(-diag_len * cos_a, -diag_len * sin_a, 'Avers.  ', ha='right', va='top', fontsize=10)

        ax.set_xlim(-lim * 1.7, lim * 1.7)
        ax.set_ylim(-lim * 2.0, lim * 1.6)  # more room at bottom for axis indicator

        # Axis indicator
        _add_axis_indicator(ax, ['State', 'Modality', 'Valence'])


def _setup_figure() -> Tuple[plt.Figure, GridSpec]:
    """Set up the figure and grid layout."""
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    
    fig = plt.figure(figsize=(16, 15))
    
    gs = GridSpec(3, 4, figure=fig, 
                     left=0.08, right=0.98,
                     bottom=0.17, top=0.93,
                     wspace=0.15, hspace=0.25,
                     width_ratios=[1, 1, 1, 0.8])
    
    return fig, gs


def _draw_legend_panel(fig, styles):
    """
    Draw a horizontal legend panel at the bottom of the figure.
    
    Layout:  All 4 rows (header, Fed, Starved, footer) span the full figure width.
    8 symbol columns evenly distributed, footer centered below.
    """
    # ── Separator line ──
    line_y = 0.13
    fig.add_artist(plt.Line2D([0.05, 0.97], [line_y, line_y],
                              transform=fig.transFigure, color='0.75',
                              linewidth=0.8, alpha=0.5, zorder=0))

    # Legend axes: full width, sits well below separator
    ax = fig.add_axes([0.03, 0.02, 0.95, 0.095])  # [left, bottom, width, height]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # ── Column definitions ──
    columns = [
        ('T app',   'fed_taste_positive',       'starved_taste_positive'),
        ('T avr',   'fed_taste_negative',        'starved_taste_negative'),
        ('O app',   'fed_odor_positive',         'starved_odor_positive'),
        ('O avr',   'fed_odor_negative',         'starved_odor_negative'),
        ('OT app',  'fed_odor_pos_taste_pos',    'starved_odor_pos_taste_pos'),
        ('T$^{+}$O$^{-}$', 'fed_odor_neg_taste_pos', 'starved_odor_neg_taste_pos'),
        ('T$^{-}$O$^{+}$', 'fed_odor_pos_taste_neg', 'starved_odor_pos_taste_neg'),
        ('OT avr',  'fed_odor_neg_taste_neg',    'starved_odor_neg_taste_neg'),
    ]

    ncols = len(columns)
    # Full-width span for all 8 symbol columns
    x_start, x_end = 0.08, 0.92
    xs = np.linspace(x_start, x_end, ncols)

    # y positions — packed tight
    y_header = 0.95
    y_fed    = 0.65
    y_stv    = 0.35
    y_footer = 0.05
    ms = 14

    # Row labels (left of first column)
    ax.text(x_start - 0.045, y_fed, 'Fed', ha='right', va='center', fontsize=9, weight='bold')
    ax.text(x_start - 0.045, y_stv, 'Stv', ha='right', va='center', fontsize=9, weight='bold')

    for i, (header, fed_key, stv_key) in enumerate(columns):
        x = xs[i]

        # Header — bold, dark
        ax.text(x, y_header, header, ha='center', va='center',
                fontsize=9, weight='bold', color='0.2')

        fed_s = styles[fed_key]
        stv_s = styles[stv_key]

        # ── Draw Fed symbol ──
        if fed_s.get('bicolor', False):
            ax.plot(x, y_fed, marker=HALF_CIRCLE_LEFT, ms=ms,
                    markerfacecolor=fed_s['left_color'], markeredgecolor=fed_s['left_edgecolor'],
                    markeredgewidth=1.2, clip_on=False, zorder=5)
            ax.plot(x, y_fed, marker=HALF_CIRCLE_RIGHT, ms=ms,
                    markerfacecolor=fed_s['right_color'], markeredgecolor=fed_s['right_edgecolor'],
                    markeredgewidth=1.2, clip_on=False, zorder=5)
        else:
            ax.plot(x, y_fed, 'o', ms=ms,
                    markerfacecolor=fed_s['color'], markeredgecolor=fed_s['edgecolor'],
                    markeredgewidth=1.2, clip_on=False, zorder=5)

        # ── Draw Starved symbol — thicker edges ──
        stv_ew = 2.5
        if stv_s.get('bicolor', False):
            ax.plot(x, y_stv, marker=HALF_CIRCLE_LEFT, ms=ms,
                    markerfacecolor=stv_s['left_color'], markeredgecolor=stv_s['left_edgecolor'],
                    markeredgewidth=stv_ew, clip_on=False, zorder=5)
            ax.plot(x, y_stv, marker=HALF_CIRCLE_RIGHT, ms=ms,
                    markerfacecolor=stv_s['right_color'], markeredgecolor=stv_s['right_edgecolor'],
                    markeredgewidth=stv_ew, clip_on=False, zorder=5)
        else:
            ax.plot(x, y_stv, 'o', ms=ms,
                    markerfacecolor=stv_s['color'], markeredgecolor=stv_s['edgecolor'],
                    markeredgewidth=stv_ew, clip_on=False, zorder=5)

    # ── Footer row: encoding rules + modality swatches, evenly spaced across full width ──
    footer_parts = [
        ('text',   dict(s=u'\u25cf  Filled = Fed',       color='0.4')),
        ('text',   dict(s=u'\u25cb  Open = Starved',     color='0.4')),
        ('text',   dict(s=u'\u25d1  Split = Conflict',   color='0.4')),
        ('swatch', dict(fc=styles['taste']['color'],      label='Taste')),
        ('swatch', dict(fc=styles['odor']['color'],       label='Odor')),
        ('swatch', dict(fc=styles['odor_taste']['color'], label='O+T')),
    ]
    n_footer = len(footer_parts)
    fxs = np.linspace(x_start, x_end, n_footer)

    for fx, (ftype, fkw) in zip(fxs, footer_parts):
        if ftype == 'text':
            ax.text(fx, y_footer, fkw['s'], ha='center', va='center',
                    fontsize=8.5, color=fkw['color'])
        elif ftype == 'swatch':
            ax.plot(fx - 0.015, y_footer, 's', ms=10, markerfacecolor=fkw['fc'],
                    markeredgecolor=fkw['fc'], clip_on=False)
            ax.text(fx + 0.01, y_footer, fkw['label'], ha='left', va='center',
                    fontsize=8.5, color='0.3')


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
        pos_inds = [i for i, n in enumerate(class_names) if '$^{+}$' in n and '$^{-}$' not in n]
        neg_inds = [i for i, n in enumerate(class_names) if '$^{-}$' in n and '$^{+}$' not in n]

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
    fixed_lim: Optional[float] = None,  # Fixed axis limit for consistent scaling across rows
) -> None:
    """
    Plot displacement vectors from the global mean to task-defined group centroids
    in a 2D latent space (e.g., t-SNE). Re-centers so the global mean is at (0, 0).
    latent: (n_samples, 2); labels: (n_samples,)
    fixed_lim: if provided, use this as the axis limit (±fixed_lim) instead of auto-computing
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

    # Set axis limits: use fixed_lim if provided, otherwise auto-compute based on vector norms
    if fixed_lim is not None:
        R = fixed_lim
    elif centroids:
        R = np.max([np.hypot(*v) for v in centroids.values()])
        R = 1.1 * R if R > 0 else 1.0
    else:
        R = 1.0
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
    best_dim: Optional[int] = None,  # dimension from best run filename (e.g., H8 -> 8)
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
                # Support both schemas: new (top-level "accuracy") and old ("data" dict)
                acc = None
                if "accuracy" in run and isinstance(run["accuracy"], (int, float)):
                    # New schema: run["accuracy"] is a float
                    acc = float(run["accuracy"])
                else:
                    # Old schema: check run["data"] dict
                    d = run.get('data', {}) or {}
                    acc = (
                        d.get('test_accuracy', None)
                        if d.get('test_accuracy', None) is not None else
                        d.get('accuracy', None)
                        if d.get('accuracy', None) is not None else
                        (d.get('metrics', {}) or {}).get('test_accuracy', None)
                    )
                    if acc is not None:
                        acc = float(acc)
                if acc is not None:
                    accs.append(acc)
            # Debug logging when acc list is empty
            if not accs:
                print(f"[DEBUG] _extract_family_stats: Empty acc list - task={task_name}, family_prefix={family_prefix}, dim_key={dim_key}, n_runs={len(runs)}, n_acc_found=0")
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
        grid_dims = [4, 8, 16, 32, 64]
    ix = {d: i for i, d in enumerate(grid_dims)}  # dim -> x position on the fixed grid

    # Star marker configuration knobs
    # Use best_dim from the best run filename if provided, otherwise default to 16
    TARGET_DIM = best_dim if best_dim is not None else 16
    REQUIRE_BOTH_FAMILIES = False      # if False: star primary at TARGET_DIM even if other family missing
    STAR_STAT = "best_accuracy"       # "best_accuracy" or "median"

    # --- primary family ---
    dims_p, stats_p = _extract_family_stats(primary_family)
    if not dims_p:
        ax.axis('off'); ax.text(0.5, 0.5, f'No {primary_family}* runs', ha='center', va='center', fontsize=10)
        return

    # optional alt - always compute stats_a for star logic, even if not plotting
    alt_family = "H" if primary_family == "E" else "E"
    dims_a, stats_a = _extract_family_stats(alt_family)

    # map dims to fixed positions (skip dims not in grid)
    dims_p = [d for d in dims_p if d in ix]
    x_p = np.array([ix[d] for d in dims_p], dtype=int)
    med_p = [stats_p[d]['median'] for d in dims_p]
    mean_p = [stats_p[d]['mean'] for d in dims_p]
    q_p = [stats_p[d]['quantiles'] for d in dims_p]
    best_p = [stats_p[d]['best_accuracy'] for d in dims_p]
    n_p = [stats_p[d]['n'] for d in dims_p]
    qlow_p = np.array([q[0] for q in q_p])
    qupp_p = np.array([q[1] for q in q_p])

    # Target dim star logic
    has_p = TARGET_DIM in stats_p and TARGET_DIM in ix
    has_a = TARGET_DIM in stats_a and TARGET_DIM in ix
    if has_p and (not REQUIRE_BOTH_FAMILIES or has_a):
        star_x = ix[TARGET_DIM]
        star_y = stats_p[TARGET_DIM][STAR_STAT]
    else:
        star_x = None
        star_y = None

    if overlay_alt_family and dims_a:
        dims_a = [d for d in dims_a if d in ix]
        if dims_a:
            x_a   = np.array([ix[d] for d in dims_a], dtype=int)
            mean_a = [stats_a[d]['mean']     for d in dims_a]
            med_a = [stats_a[d]['median']     for d in dims_a]
            q_a   = [stats_a[d]['quantiles']  for d in dims_a]
            qlow_a = np.array([q[0] for q in q_a])
            qupp_a = np.array([q[1] for q in q_a])

            # Error bars for alt
            err_low_a = np.array([med_a[i] - qlow_a[i] for i in range(len(med_a))])
            err_up_a = np.array([qupp_a[i] - med_a[i] for i in range(len(med_a))])

            ax.plot(
                x_a, med_a, 'o-', color='darkcyan',markersize=8, linewidth=1.5, alpha = 0.8,
                markerfacecolor='white', markeredgecolor='darkcyan', markeredgewidth=1.5,
                label=f"{alt_family}", zorder=2
            )
            ax.fill_between(x_a, qlow_a, qupp_a, alpha=0.3, color='darkcyan', linewidth=0, zorder=1)
            ax.errorbar(x_a, med_a, yerr=[err_low_a, err_up_a], fmt='none', capsize=2, 
                       elinewidth=1, alpha=0.9, zorder=3, color='darkcyan')

    # draw primary
    # Error bars for primary
    err_low_p = np.array([med_p[i] - qlow_p[i] for i in range(len(med_p))])
    err_up_p = np.array([qupp_p[i] - med_p[i] for i in range(len(med_p))])

    ax.plot(x_p, med_p, 'o-', color='k', markersize=8, linewidth=1.5, alpha = 0.8,
            markerfacecolor='white', markeredgecolor='k', markeredgewidth=1.5,label=f"{primary_family}", zorder=2)
    ax.fill_between(x_p, qlow_p, qupp_p, alpha=0.5, color='gray', linewidth=0, zorder=1)
    ax.errorbar(x_p, med_p, yerr=[err_low_p, err_up_p], fmt='none', capsize=2, 
               elinewidth=1, alpha=0.9, zorder=3, color='k')
    
    # Plot star if conditions met
    if star_x is not None and star_y is not None:
        ax.plot(star_x, star_y, marker='*', markersize=10, color='gold',
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
    if overlay_alt_family:
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
    bicolor_info: Dict = None,
    use_l_axis: bool = False,
    is_control: bool = False,
    plot_centroids: bool = False,
    task: str = None,
    logger = None,
    centroid_fixed_lim: Optional[float] = None,  # Fixed axis limit for centroid plots
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
        bicolor_info: Dict mapping class index to bicolor style info
        use_l_axis: Use L-shaped axis
        is_control: If control plot
        plot_centroids: Add centroid vectors
        task: Task name for centroids
        centroid_fixed_lim: Fixed axis limit for centroid plots (±lim)
    """
    # Get data
    data_key = 'control' if is_control else \
              next((k for k in ['best', 'control'] if k in run_dict),
                  next(k for k in run_dict if k != '__class_names__'))
    if logger:
        logger.debug(f"_plot_tsne: Accessing data with key '{data_key}' for task '{task}' (row={row}, col={col})")
    data = run_dict[data_key]
    if logger:
        logger.debug(f"_plot_tsne: Data type: {type(data)}, has 'tsne_2d': {'tsne_2d' in data if isinstance(data, dict) else 'N/A'}, has 'latent_labels': {'latent_labels' in data if isinstance(data, dict) else 'N/A'}")
    
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
            bicolor_info=bicolor_info,
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
            styles=styles,
            fixed_lim=centroid_fixed_lim,
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
    bicolor_info: Dict = None,
    out_path: str = 'results/CombiPlots/fig_latent_chptRuns_1.png',
    use_l_axis: bool = True,
    logger = None
) -> None:
   
    """Generate the complete figure with t-SNE plots and accuracy vs dimension."""
    # Setup
    fig, gs = _setup_figure()
    
    tasks = list(results_dict.keys())[:3]
    task_names = ['i. State', 'ii. State, Modality', 'iii. State, Modality, Valence']
    task_y_pos = [0.81, 0.53, 0.25]
    
    column_titles = ['a. Control t-SNE', 'b. Model t-SNE', 'c. Centroid projections', 'd. Accuracy']
    column_positions = [0.18, 0.43, 0.65, 0.9] 

    for x, title in zip(column_positions, column_titles):
        fig.text(x, 0.95, title, ha='center', va='center', fontsize=12, weight='bold')
    
    # Plot each task
    for row, (task, task_name) in enumerate(zip(tasks, task_names)):
        if logger:
            logger.debug(f"Plotting task: {task} (row {row})")
        run_dict = results_dict[task]
        class_names = run_dict.get('__class_names__')
        if logger:
            logger.debug(f"Task '{task}': class_names = {class_names}")
        
        # Add row label
        fig.text(0.05, task_y_pos[row], task_name, ha='left', va='center', fontsize=12, rotation=90, weight='bold', transform=fig.transFigure)
        
        # Bicolor info for this task
        task_bicolor = bicolor_info.get(task, {}) if bicolor_info else {}

        # Plot control t-SNE (first column)
        _plot_tsne(fig, gs, row, 0, run_dict, class_names, colors[task], edges[task], shapes[task], bicolor_info=task_bicolor, use_l_axis=use_l_axis, is_control=True, task=task, logger=logger)
        
        # Plot best t-SNE (second column)
        _plot_tsne(fig, gs, row, 1, run_dict, class_names, colors[task], edges[task], shapes[task], bicolor_info=task_bicolor, use_l_axis=use_l_axis, task=task, logger=logger)
        
        # ── NEW: Plot biological axes (third column) ──
        # Uses full high-D latent space, NOT t-SNE 2D
        best_data = run_dict.get('best', {})
        X_hd = best_data.get('transformer_latent_space')
        latent_labels = best_data.get('latent_labels')
        
        if X_hd is not None and latent_labels is not None:
            ax_bio = fig.add_subplot(gs[row, 2])
            plot_biological_axes_panel(
                ax=ax_bio,
                X=X_hd,
                labels=latent_labels,
                class_names=class_names,
                task=task,
                colors=colors[task],
                edges=edges[task],
                bicolor_info=task_bicolor,
                fig=fig,
                gs=gs,
                row=row,
            )
        else:
            # Fallback: empty panel with warning
            ax_bio = fig.add_subplot(gs[row, 2])
            ax_bio.text(0.5, 0.5, 'No high-D data', ha='center', va='center',
                       fontsize=10, color='red', transform=ax_bio.transAxes)
            ax_bio.set_xticks([]); ax_bio.set_yticks([])
            if logger:
                logger.warning(f"Task '{task}': missing transformer_latent_space for biological axes")
        
        # Plot accuracy vs dimension for each task
        ax = fig.add_subplot(gs[row, 3])
        # Extract best dimension based on primary family (H* -> trf_dim, E* -> cnn_dim)
        primary_family = "H"
        if primary_family == "H":
            best_dim = run_dict.get('best_trf_dim')
        else:
            best_dim = run_dict.get('best_cnn_dim')
        _plot_accuracy_vs_dimension(ax, run_dict, task_name, color=list(colors.values())[row], row=row, 
                                    primary_family=primary_family, overlay_alt_family=False, best_dim=best_dim)
    
    # Draw legend panel at bottom of figure
    _draw_legend_panel(fig, styles)

    # Save figure
    if logger:
        logger.debug(f"Saving figure to {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.savefig(out_path.replace('.png', '.svg'), dpi=300, format='svg', bbox_inches='tight', pad_inches=0.15)
    logger.info(f"Saved SVG : {out_path.replace('.png', '.svg')}")
    plt.close()
    if logger:
        logger.info(f"Figure saved to {out_path}")
    else:
        print(f"Figure saved to {out_path}")



def main():
    logger = setup_logger(task_name="run_figure_hLatent", log_dir="logs/run_figure_hLatent")
    logger.info("Starting main")

    # BASE_RESULTS_DIR = os.path.join('results', '_archive', '_SeedFinals')
    BASE_RESULTS_DIR = os.path.join('results', '_chkpt_finals')
    logger.info(f"BASE_RESULTS_DIR: {BASE_RESULTS_DIR}")
    logger.debug(f"Checking if BASE_RESULTS_DIR exists: {os.path.exists(BASE_RESULTS_DIR)}")
    
    logger.debug("Loading styles from YAML file...")
    styles, TASK_CLASS_NAMES, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES, TASK_BICOLOR_INFO = get_style(style="stylesG")
    logger.debug(f"Loaded styles. TASK_CLASS_NAMES keys: {list(TASK_CLASS_NAMES.keys()) if isinstance(TASK_CLASS_NAMES, dict) else 'N/A'}")

    logger.debug(f"Loading results from {BASE_RESULTS_DIR}...")
    logger.debug(f"Task names to load: {list(TASK_CLASS_NAMES.keys()) if isinstance(TASK_CLASS_NAMES, dict) else TASK_CLASS_NAMES}")
    # only_cnn_dim=16: skip all runs where E != 16 (saves significant load time)
    results_dict = load_all_results(BASE_RESULTS_DIR, TASK_CLASS_NAMES, fixed_trf_for_E=16, fixed_cnn_for_H=16, only_cnn_dim=16, logger=logger)
    logger.debug(f"Loaded results. Tasks found: {list(results_dict.keys())}")
    
    # Debug: Check what was loaded for each task
    for task_name, task_data in results_dict.items():
        logger.debug(f"Task '{task_name}':")
        logger.debug(f"  - Has control: {task_data.get('control') is not None}")
        logger.debug(f"  - Has best: {task_data.get('best') is not None}")
        logger.debug(f"  - Run groups: {list(task_data.get('runs', {}).keys())}")
        for group_key, runs in task_data.get('runs', {}).items():
            logger.debug(f"    - Group {group_key}: {len(runs)} runs")

    logger.info("Loaded results successfully")

    logger.debug("Generating figure...")
    plot_figure_latent(results_dict, styles, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES, bicolor_info=TASK_BICOLOR_INFO, out_path='results/CombiPlots/fig_Latent_v13.png', logger=logger)
    logger.info("Figure generation complete")

if __name__ == '__main__':
    main()