"""
Centroid Visualization Script — All 3 Tasks
============================================
Version A: Biological axes (1D → 2D → 3D)
Version B: PCA group arrows
Version C: 2D panels (biological axes, all pairs)
Version D: Pairwise Euclidean distance & cosine similarity matrices (16D, lossless)
+ PCA variance diagnostics
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from mpl_toolkits.mplot3d import Axes3D
from src.utils.helpers import get_style


# ═══════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════

RESULT_PATHS = {
    'MetabolicState_2':          "/rhomes/aabdel/DrosoEmbedding/results/MetabolicState_2_C2_E16_H16_12/evaluation/C2_E16_H16_12_evalResults.pkl",
    'State_Modality_6':          "/rhomes/aabdel/DrosoEmbedding/results/State_Modality_6_C6_E16_H16_34/evaluation/C6_E16_H16_34_evalResults.pkl",
    'State_Modality_Valence_16': "/rhomes/aabdel/DrosoEmbedding/results/State_Modality_Valence_16_C16_E16_H16_42/evaluation/C16_E16_H16_42_evalResults.pkl",
}

# OUT_DIR = "/rhomes/aabdel/DrosoEmbedding/results/CombiPlots/"


# ═══════════════════════════════════════════════
# DATA SETUP
# ═══════════════════════════════════════════════

def load_results(result_paths):
    """Load pickle files for all tasks."""
    results = {}
    for task_key, path in result_paths.items():
        with open(path, 'rb') as f:
            results[task_key] = pickle.load(f)
        print(f"Loaded {task_key} from {path}")
    return results


def setup_data(results, TASK_CLASS_NAMES, TASK_COLORS, TASK_EDGECOLORS):
    """Organize all task data into a single dict."""
    tasks = {}
    for task_key in results:
        tasks[task_key] = {
            'X': results[task_key]['transformer_latent_space'],
            'labels': results[task_key]['latent_labels'],
            'class_names': TASK_CLASS_NAMES[task_key],
            'colors': TASK_COLORS[task_key],
            'edges': TASK_EDGECOLORS[task_key],
        }
    return tasks


# ═══════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════

def get_group_map(task_key, class_names):
    """Group map for PCA arrow plots."""
    if task_key == 'MetabolicState_2':
        return {'S': [0], 'F': [1]}
    if task_key == 'State_Modality_6':
        return {
            'S': [0, 2, 4], 'F': [1, 3, 5],
            'O': [0, 1], 'T': [2, 3], 'O/T': [4, 5],
        }
    if task_key == 'State_Modality_Valence_16':
        pos_inds = [i for i, n in enumerate(class_names) if '$^{+}$' in n and '$^{-}$' not in n]
        neg_inds = [i for i, n in enumerate(class_names) if '$^{-}$' in n and '$^{+}$' not in n]
        return {
            'S': [0, 1, 4, 5, 8, 9, 10, 11],
            'F': [2, 3, 6, 7, 12, 13, 14, 15],
            'O': [0, 1, 2, 3], 'T': [4, 5, 6, 7],
            'O/T': [8, 9, 10, 11, 12, 13],
            '+': pos_inds, '-': neg_inds,
            '+/-': [10, 11, 14, 15],
        }
    return {}


def get_group_color(name, styles=None):
    """Color mapping for group arrows."""
    default_colors = {
        'S': '#808080', 'F': '#595959',
        'O': '#D35F2A', 'T': '#3382BE', 'O/T': '#7B4278',
        '+': '#2CA02C', '-': '#8B4513', '+/-': '#9467BD',
    }
    if styles:
        key_map = {
            'S': 'starved', 'F': 'fed', 'O': 'odor', 'T': 'taste',
            'O/T': 'odor_taste', '+': 'positive', '-': 'negative', '+/-': 'positive_negative',
        }
        style_key = key_map.get(name)
        if style_key and style_key in styles:
            d = styles[style_key]
            return d.get('arrow_color', d.get('color', default_colors.get(name, 'k')))
    return default_colors.get(name, 'k')


def compute_biological_axes(X, labels, task_key, class_names):
    """Compute biological axes and project class centroids."""
    global_mu = X.mean(axis=0)
    unique_labels = np.sort(np.unique(labels))
    centroids = np.array([X[labels == l].mean(axis=0) for l in unique_labels])
    displacements = centroids - global_mu

    if task_key == 'MetabolicState_2':
        mean_S = X[labels == 0].mean(axis=0)
        mean_F = X[labels == 1].mean(axis=0)
        state_axis = mean_F - mean_S
        state_axis = state_axis / np.linalg.norm(state_axis)
        proj_state = displacements @ state_axis
        return {'state': proj_state}, {'state': state_axis}, displacements

    if task_key == 'State_Modality_6':
        mean_S = X[np.isin(labels, [0, 2, 4])].mean(axis=0)
        mean_F = X[np.isin(labels, [1, 3, 5])].mean(axis=0)
        mean_O = X[np.isin(labels, [0, 1])].mean(axis=0)
        mean_T = X[np.isin(labels, [2, 3])].mean(axis=0)

        state_axis = mean_F - mean_S
        state_axis = state_axis / np.linalg.norm(state_axis)
        modality_axis = mean_T - mean_O
        modality_axis = modality_axis / np.linalg.norm(modality_axis)

        proj_state = displacements @ state_axis
        proj_modality = displacements @ modality_axis

        print(f"  6-class: State · Modality = {np.dot(state_axis, modality_axis):.3f}")
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
        state_axis = state_axis / np.linalg.norm(state_axis)
        modality_axis = mean_T - mean_O
        modality_axis = modality_axis / np.linalg.norm(modality_axis)
        valence_axis = mean_pos - mean_neg
        valence_axis = valence_axis / np.linalg.norm(valence_axis)

        proj_state = displacements @ state_axis
        proj_modality = displacements @ modality_axis
        proj_valence = displacements @ valence_axis

        print(f"  16-class: State · Modality   = {np.dot(state_axis, modality_axis):.3f}")
        print(f"  16-class: State · Valence    = {np.dot(state_axis, valence_axis):.3f}")
        print(f"  16-class: Modality · Valence = {np.dot(modality_axis, valence_axis):.3f}")

        return {'state': proj_state, 'modality': proj_modality, 'valence': proj_valence}, \
               {'state': state_axis, 'modality': modality_axis, 'valence': valence_axis}, displacements


def short_name(n):
    """Shorten LaTeX class names for plot labels."""
    return n.replace('$^{+}$', '+').replace('$^{-}$', '-').replace(' (S)', ' S').replace(' (F)', ' F')


def _save_or_show(fig, save_path):
    """Save figure to path or show interactively."""
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")
    else:
        plt.show()


# ═══════════════════════════════════════════════
# VERSION A: Biological Axes (1D → 2D → 3D)
# ═══════════════════════════════════════════════

def plot_version_A(tasks, save_path=None):
    """
    Row i:   2-class  → 1D number line (state)
    Row ii:  6-class  → 2D scatter (state × modality)
    Row iii: 16-class → 3D scatter (state × modality × valence)
    """
    fig = plt.figure(figsize=(8, 14))
    task_keys = ['MetabolicState_2', 'State_Modality_6', 'State_Modality_Valence_16']
    row_labels = ['i. State', 'ii. State, Modality', 'iii. State, Modality, Valence']

    for row, (task_key, row_label) in enumerate(zip(task_keys, row_labels)):
        t = tasks[task_key]
        X, labels, class_names = t['X'], t['labels'], t['class_names']
        colors, edges = t['colors'], t['edges']
        projections, axes_dict, displacements = compute_biological_axes(X, labels, task_key, class_names)
        n_classes = len(np.unique(labels))

        total_var = np.sum(displacements ** 2)
        explained = sum(np.sum(p ** 2) for p in projections.values())
        pct = explained / total_var * 100

        if task_key == 'MetabolicState_2':
            ax = fig.add_subplot(3, 1, row + 1)
            proj_s = projections['state']
            for i in range(n_classes):
                is_starved = '(S)' in class_names[i] or class_names[i] == 'Starved'
                ax.scatter(proj_s[i], 0, c=colors[i], edgecolors=edges[i],
                           linewidth=2.0 if is_starved else 0.5, s=200, zorder=3)
                ax.annotate(short_name(class_names[i]), (proj_s[i], 0),
                            xytext=(0, 15), textcoords='offset points', ha='center', fontsize=10)
            ax.axhline(0, lw=1, alpha=0.3, color='gray')
            ax.plot(0, 0, 'k+', ms=10, mew=2, zorder=5)
            ax.set_xlabel('← Starved | Fed →', fontsize=11)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_title(f'{row_label}  ({pct:.1f}% var.)', fontsize=12, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

        elif task_key == 'State_Modality_6':
            ax = fig.add_subplot(3, 1, row + 1)
            proj_s, proj_m = projections['state'], projections['modality']
            for i in range(n_classes):
                is_starved = '(S)' in class_names[i]
                ax.scatter(proj_s[i], proj_m[i], c=colors[i], edgecolors=edges[i],
                           linewidth=2.0 if is_starved else 0.5, s=200, zorder=3)
                ax.annotate(short_name(class_names[i]), (proj_s[i], proj_m[i]),
                            xytext=(6, 6), textcoords='offset points', ha='left', fontsize=9)
            ax.axhline(0, lw=0.5, alpha=0.2, color='gray')
            ax.axvline(0, lw=0.5, alpha=0.2, color='gray')
            ax.plot(0, 0, 'k+', ms=10, mew=2, zorder=5)
            ax.set_xlabel('← Starved | Fed →', fontsize=11)
            ax.set_ylabel('← Odor | Taste →', fontsize=11)
            ax.set_aspect('equal')
            ax.set_title(f'{row_label}  ({pct:.1f}% var.)', fontsize=12, fontweight='bold')

        elif task_key == 'State_Modality_Valence_16':
            ax = fig.add_subplot(3, 1, row + 1, projection='3d')
            proj_s, proj_m, proj_v = projections['state'], projections['modality'], projections['valence']
            lim = 2.2
            ax.plot([-lim, lim], [0, 0], [0, 0], 'k-', lw=1.5, alpha=0.4)
            ax.plot([0, 0], [-lim, lim], [0, 0], 'k-', lw=1.5, alpha=0.4)
            ax.plot([0, 0], [0, 0], [-lim, lim], 'k-', lw=1.5, alpha=0.4)
            ax.text(lim, 0, 0, '  Fed →', fontsize=9, fontweight='bold')
            ax.text(-lim, 0, 0, '← Starved', fontsize=9, fontweight='bold')
            ax.text(0, lim, 0, '  Taste →', fontsize=9, fontweight='bold')
            ax.text(0, -lim, 0, '← Odor', fontsize=9, fontweight='bold')
            ax.text(0, 0, lim, '  Appetitive →', fontsize=9, fontweight='bold')
            ax.text(0, 0, -lim, '← Aversive', fontsize=9, fontweight='bold')
            for i in range(n_classes):
                is_starved = '(S)' in class_names[i]
                ax.scatter(proj_s[i], proj_m[i], proj_v[i], c=colors[i], edgecolors=edges[i],
                           linewidth=2.0 if is_starved else 0.5, s=150, depthshade=False)
            ax.scatter([0], [0], [0], c='red', s=80, marker='+', linewidths=3, zorder=100)
            ax.grid(False)
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis.pane.fill = False
                axis.pane.set_edgecolor('none')
                axis.line.set_color('none')
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
            ax.view_init(elev=25, azim=-45)
            ax.set_title(f'{row_label}  ({pct:.1f}% var.)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ═══════════════════════════════════════════════
# VERSION B: PCA Group Arrows
# ═══════════════════════════════════════════════

def plot_version_B(tasks, styles=None, save_path=None):
    """Same layout for all 3 rows: PCA-projected group displacement arrows."""
    fig, axes_arr = plt.subplots(3, 1, figsize=(7, 14))
    task_keys = ['MetabolicState_2', 'State_Modality_6', 'State_Modality_Valence_16']
    row_labels = ['i. State', 'ii. State, Modality', 'iii. State, Modality, Valence']

    for row, (task_key, row_label) in enumerate(zip(task_keys, row_labels)):
        ax = axes_arr[row]
        t = tasks[task_key]
        X, labels, class_names = t['X'], t['labels'], t['class_names']
        global_mu = X.mean(axis=0)
        group_map = get_group_map(task_key, class_names)

        centroids_hd = {}
        for name, inds in group_map.items():
            mask = np.isin(labels, inds)
            if np.any(mask):
                centroids_hd[name] = X[mask].mean(axis=0) - global_mu

        names = list(centroids_hd.keys())
        vecs = np.array([centroids_hd[n] for n in names])

        if len(names) <= 2:
            pca = PCA(n_components=min(2, len(names))).fit(vecs)
            vecs_2d = pca.transform(vecs)
            if vecs_2d.shape[1] == 1:
                vecs_2d = np.column_stack([vecs_2d, np.zeros(len(names))])
            total_var = sum(pca.explained_variance_ratio_)
        else:
            pca = PCA(n_components=2).fit(vecs)
            vecs_2d = pca.transform(vecs)
            total_var = sum(pca.explained_variance_ratio_[:2])

        for i, name in enumerate(names):
            dx, dy = vecs_2d[i]
            col = get_group_color(name, styles)
            ax.arrow(0, 0, dx, dy, head_width=0.05, head_length=0.03, linewidth=2.5,
                     alpha=0.7, color=col, length_includes_head=True)
            ax.annotate(name, xy=(dx * 0.6, dy * 0.6), xytext=(0, 10),
                        textcoords='offset points', ha='center', fontsize=13,
                        fontweight='bold', color=col)

        ax.axhline(0, lw=0.5, alpha=0.2, color='gray')
        ax.axvline(0, lw=0.5, alpha=0.2, color='gray')
        ax.plot(0, 0, 'ko', ms=4, zorder=10)
        ax.set_aspect('equal')
        R = 1.3 * np.max(np.abs(vecs_2d)) if len(vecs_2d) > 0 else 1.0
        ax.set_xlim(-R, R); ax.set_ylim(-R, R)
        if pca.n_components_ >= 2:
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.0%})', fontsize=9)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.0%})', fontsize=9)
        ax.set_title(f'{row_label}  (PCA: {total_var:.0%} var.)', fontsize=12, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ═══════════════════════════════════════════════
# VERSION C: 2D Panels (biological axes, all pairs)
# ═══════════════════════════════════════════════

def plot_version_C(tasks, save_path=None):
    """
    Row i:   1 panel  (state only → 1D strip)
    Row ii:  1 panel  (state × modality)
    Row iii: 3 panels (state×modality, state×valence, modality×valence)
    """
    fig, axes_grid = plt.subplots(3, 3, figsize=(15, 13))
    task_keys = ['MetabolicState_2', 'State_Modality_6', 'State_Modality_Valence_16']
    row_labels = ['i. State', 'ii. State, Modality', 'iii. State, Modality, Valence']

    for row, (task_key, row_label) in enumerate(zip(task_keys, row_labels)):
        t = tasks[task_key]
        X, labels, class_names = t['X'], t['labels'], t['class_names']
        colors, edges = t['colors'], t['edges']
        n_classes = len(np.unique(labels))
        projections, axes_dict, displacements = compute_biological_axes(X, labels, task_key, class_names)

        if task_key == 'MetabolicState_2':
            axes_grid[row, 1].set_visible(False)
            axes_grid[row, 2].set_visible(False)
            ax = axes_grid[row, 0]
            proj_s = projections['state']
            for i in range(n_classes):
                is_starved = '(S)' in class_names[i] or class_names[i] == 'Starved'
                ax.scatter(proj_s[i], 0, c=colors[i], edgecolors=edges[i],
                           linewidth=2.0 if is_starved else 0.5, s=200, zorder=3)
                ax.annotate(short_name(class_names[i]), (proj_s[i], 0),
                            xytext=(0, 15), textcoords='offset points', ha='center', fontsize=10)
            ax.axhline(0, lw=0.5, alpha=0.2, color='gray')
            ax.plot(0, 0, 'k+', ms=10, mew=2, zorder=5)
            ax.set_xlabel('← Starved | Fed →', fontsize=10)
            ax.set_ylim(-0.5, 0.5); ax.set_yticks([])
            ax.set_title(f'{row_label}', fontsize=11, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

        elif task_key == 'State_Modality_6':
            axes_grid[row, 1].set_visible(False)
            axes_grid[row, 2].set_visible(False)
            ax = axes_grid[row, 0]
            proj_s, proj_m = projections['state'], projections['modality']
            for i in range(n_classes):
                is_starved = '(S)' in class_names[i]
                ax.scatter(proj_s[i], proj_m[i], c=colors[i], edgecolors=edges[i],
                           linewidth=2.0 if is_starved else 0.5, s=200, zorder=3)
                ax.annotate(short_name(class_names[i]), (proj_s[i], proj_m[i]),
                            xytext=(6, 6), textcoords='offset points', ha='left', fontsize=9)
            ax.axhline(0, lw=0.5, alpha=0.2, color='gray')
            ax.axvline(0, lw=0.5, alpha=0.2, color='gray')
            ax.plot(0, 0, 'k+', ms=10, mew=2, zorder=5)
            ax.set_xlabel('← Starved | Fed →', fontsize=10)
            ax.set_ylabel('← Odor | Taste →', fontsize=10)
            ax.set_aspect('equal')
            ax.set_title(f'{row_label}', fontsize=11, fontweight='bold')

        elif task_key == 'State_Modality_Valence_16':
            proj_s, proj_m, proj_v = projections['state'], projections['modality'], projections['valence']
            pairs = [
                (proj_s, proj_m, '← Starved | Fed →', '← Odor | Taste →'),
                (proj_s, proj_v, '← Starved | Fed →', '← Aversive | Appetitive →'),
                (proj_m, proj_v, '← Odor | Taste →', '← Aversive | Appetitive →'),
            ]
            for col, (px, py, xlabel, ylabel) in enumerate(pairs):
                ax = axes_grid[row, col]
                for i in range(n_classes):
                    is_starved = '(S)' in class_names[i]
                    ax.scatter(px[i], py[i], c=colors[i], edgecolors=edges[i],
                               linewidth=2.0 if is_starved else 0.5, s=150, zorder=3)
                ax.axhline(0, lw=0.5, alpha=0.2, color='gray')
                ax.axvline(0, lw=0.5, alpha=0.2, color='gray')
                ax.plot(0, 0, 'k+', ms=10, mew=2, zorder=5)
                ax.set_xlabel(xlabel, fontsize=9)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.set_aspect('equal')
                if col == 0:
                    ax.set_title(f'{row_label}', fontsize=11, fontweight='bold')

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ═══════════════════════════════════════════════
# VERSION D: Distance & Cosine Similarity Matrices
# ═══════════════════════════════════════════════

def plot_version_D(tasks, save_path=None):
    """
    For each task: pairwise Euclidean distance and cosine similarity
    between class centroids in the full high-dimensional space.
    No projection — completely lossless.
    """
    task_keys = ['MetabolicState_2', 'State_Modality_6', 'State_Modality_Valence_16']
    row_labels = ['i. State (2-class)', 'ii. State, Modality (6-class)', 'iii. State, Modality, Valence (16-class)']

    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    for row, (task_key, row_label) in enumerate(zip(task_keys, row_labels)):
        t = tasks[task_key]
        X, labels, class_names = t['X'], t['labels'], t['class_names']
        d_model = X.shape[1]

        # Per-class centroids
        unique_labels = np.sort(np.unique(labels))
        centroids = np.array([X[labels == l].mean(axis=0) for l in unique_labels])
        global_mu = X.mean(axis=0)
        displacements = centroids - global_mu

        # Short labels
        short_names = [short_name(n) for n in class_names]

        # Pairwise Euclidean distance
        dist_mat = squareform(pdist(centroids, metric='euclidean'))

        # Cosine similarity of displacement vectors
        cos_mat = cosine_similarity(displacements)

        n = len(unique_labels)

        # --- Euclidean distance ---
        ax1 = axes[row, 0]
        im1 = ax1.imshow(dist_mat, cmap='viridis')
        ax1.set_xticks(range(n)); ax1.set_xticklabels(short_names, fontsize=7, rotation=90)
        ax1.set_yticks(range(n)); ax1.set_yticklabels(short_names, fontsize=7)
        ax1.set_title(f'{row_label}\nEuclidean distance ({d_model}D)', fontsize=10, fontweight='bold')
        plt.colorbar(im1, ax=ax1, shrink=0.8)

        # --- Cosine similarity ---
        ax2 = axes[row, 1]
        im2 = ax2.imshow(cos_mat, cmap='RdBu_r', vmin=-1, vmax=1)
        ax2.set_xticks(range(n)); ax2.set_xticklabels(short_names, fontsize=7, rotation=90)
        ax2.set_yticks(range(n)); ax2.set_yticklabels(short_names, fontsize=7)
        ax2.set_title(f'{row_label}\nCosine similarity ({d_model}D)', fontsize=10, fontweight='bold')
        plt.colorbar(im2, ax=ax2, shrink=0.8)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_version_E(tasks, save_path=None):
    task_keys = ['MetabolicState_2', 'State_Modality_6', 'State_Modality_Valence_16']
    row_labels = ['i. State (2-class)', 'ii. State, Modality (6-class)', 'iii. State, Modality, Valence (16-class)']

    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    for row, (task_key, row_label) in enumerate(zip(task_keys, row_labels)):
        t = tasks[task_key]
        X, labels, class_names = t['X'], t['labels'], t['class_names']
        d_model = X.shape[1]
        global_mu = X.mean(axis=0)

        # ── USE GROUP CENTROIDS INSTEAD OF CLASS CENTROIDS ──
        group_map = get_group_map(task_key, class_names)
        group_names = list(group_map.keys())
        centroids = np.array([X[np.isin(labels, inds)].mean(axis=0) for inds in group_map.values()])
        displacements = centroids - global_mu

        # Pairwise Euclidean distance
        dist_mat = squareform(pdist(centroids, metric='euclidean'))

        # Cosine similarity of displacement vectors
        cos_mat = cosine_similarity(displacements)

        n = len(group_names)

        # --- Euclidean distance ---
        ax1 = axes[row, 0]
        im1 = ax1.imshow(dist_mat, cmap='viridis')
        ax1.set_xticks(range(n)); ax1.set_xticklabels(group_names, fontsize=9, rotation=45)
        ax1.set_yticks(range(n)); ax1.set_yticklabels(group_names, fontsize=9)
        ax1.set_title(f'{row_label}\nEuclidean distance ({d_model}D)', fontsize=10, fontweight='bold')
        plt.colorbar(im1, ax=ax1, shrink=0.8)

        # --- Cosine similarity ---
        ax2 = axes[row, 1]
        im2 = ax2.imshow(cos_mat, cmap='RdBu_r', vmin=-1, vmax=1)
        ax2.set_xticks(range(n)); ax2.set_xticklabels(group_names, fontsize=9, rotation=45)
        ax2.set_yticks(range(n)); ax2.set_yticklabels(group_names, fontsize=9)
        ax2.set_title(f'{row_label}\nCosine similarity ({d_model}D)', fontsize=10, fontweight='bold')
        plt.colorbar(im2, ax=ax2, shrink=0.8)

    plt.tight_layout()
    _save_or_show(fig, save_path)

# ═══════════════════════════════════════════════
# PCA VARIANCE DIAGNOSTICS
# ═══════════════════════════════════════════════

def print_pca_diagnostics(tasks):
    """Print PCA variance distribution for each task."""
    print("\n" + "=" * 60)
    print("PCA VARIANCE DIAGNOSTICS")
    print("=" * 60)

    for task_key, t in tasks.items():
        X = t['X']
        pca = PCA().fit(X)
        cumvar = np.cumsum(pca.explained_variance_ratio_)

        print(f"\n{task_key} | shape: {X.shape}")
        print(f"  PC1:     {pca.explained_variance_ratio_[0]:.1%}")
        print(f"  PC2:     {pca.explained_variance_ratio_[1]:.1%}")
        print(f"  PC1+PC2: {cumvar[1]:.1%}")
        print(f"  PCs for 90%: {np.searchsorted(cumvar, 0.9) + 1}")
        print(f"  PCs for 95%: {np.searchsorted(cumvar, 0.95) + 1}")

def print_biological_axes_diagnostics(tasks):
    """Print dot products, norms, and variance captured for biological axes."""
    print("\n" + "=" * 60)
    print("BIOLOGICAL AXES DIAGNOSTICS")
    print("=" * 60)

    for task_key, t in tasks.items():
        X, labels, class_names = t['X'], t['labels'], t['class_names']
        projections, axes_dict, displacements = compute_biological_axes(X, labels, task_key, class_names)

        # Variance captured
        total_var = np.sum(displacements ** 2)
        explained = sum(np.sum(p ** 2) for p in projections.values())

        # PCA optimum for same number of axes
        n_axes = len(axes_dict)
        pca_opt = PCA(n_components=min(n_axes, displacements.shape[0])).fit(displacements)
        pca_var = sum(pca_opt.explained_variance_ratio_) * total_var

        print(f"\n{task_key}")
        print(f"  Axes: {list(axes_dict.keys())}")

        # Dot products
        axis_names = list(axes_dict.keys())
        for i in range(len(axis_names)):
            for j in range(i + 1, len(axis_names)):
                dot = np.dot(axes_dict[axis_names[i]], axes_dict[axis_names[j]])
                print(f"  {axis_names[i]} · {axis_names[j]} = {dot:.3f}")

        # Group norms
        group_map = get_group_map(task_key, class_names)
        global_mu = X.mean(axis=0)
        print(f"  Group norms:")
        for name, inds in group_map.items():
            mask = np.isin(labels, inds)
            vec = X[mask].mean(axis=0) - global_mu
            print(f"    {name:4s}: {np.linalg.norm(vec):.3f}")

        # Variance
        print(f"  Biological axes variance: {explained / total_var:.1%}")
        print(f"  PCA optimum ({n_axes} axes):    {pca_var / total_var:.1%}")
        print(f"  Ratio:                    {explained / pca_var:.1%}")


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    # out_dir = OUT_DIR
    # os.makedirs(out_dir, exist_ok=True)

    # Load data
    results = load_results(RESULT_PATHS)
    styles, TASK_CLASS_NAMES, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES = get_style(style="stylesE")
    tasks = setup_data(results, TASK_CLASS_NAMES, TASK_COLORS, TASK_EDGECOLORS)

    # PCA diagnostics
    print_pca_diagnostics(tasks)
    print_biological_axes_diagnostics(tasks)

    # Generate all figures
    print("\nGenerating figures...")
    plot_version_A(tasks, save_path='/rhomes/aabdel/DrosoEmbedding/results/CombiPlots/centroidsA_biological_axes.png')
    plot_version_B(tasks, styles=styles, save_path='/rhomes/aabdel/DrosoEmbedding/results/CombiPlots/centroidsB_pca_arrows.png')
    plot_version_C(tasks, save_path='/rhomes/aabdel/DrosoEmbedding/results/CombiPlots/centroidsC_2d_panels.png')
    plot_version_D(tasks, save_path='/rhomes/aabdel/DrosoEmbedding/results/CombiPlots/centroidsD_distance_matrices.png')
    plot_version_E(tasks, save_path='/rhomes/aabdel/DrosoEmbedding/results/CombiPlots/centroidsE_GroupsDistance_matrices.png')


if __name__ == '__main__':
    main()