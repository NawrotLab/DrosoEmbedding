import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from src.utils.helpers import get_group_map
from sklearn.manifold import TSNE


def compute_centroid_tsne_payload(
    features_hd: np.ndarray,          # (N, D) == transformer_latent_space
    labels: np.ndarray,               # (N,)
    task_name: str,
    class_names: list[str],
    random_state: int = 30,
) -> dict:
    assert features_hd.ndim == 2 and labels.ndim == 1
    N, D = features_hd.shape
    assert labels.shape[0] == N

    group_map = get_group_map(task_name, class_names)

    # Build groups: grouped if map exists, else per-class
    if group_map is None:
        groups = [str(i) for i in range(len(class_names))]
        centroid_hd = []
        for cls_idx in range(len(class_names)):
            m = (labels == cls_idx)
            centroid_hd.append(features_hd[m].mean(axis=0) if np.any(m) else np.full((D,), np.nan, np.float32))
        centroid_hd = np.stack(centroid_hd, axis=0).astype(np.float32)
    else:
        groups = list(group_map.keys())
        centroid_hd = []
        for name in groups:
            inds = group_map[name]
            m = np.isin(labels, inds)
            centroid_hd.append(features_hd[m].mean(axis=0) if np.any(m) else np.full((D,), np.nan, np.float32))
        centroid_hd = np.stack(centroid_hd, axis=0).astype(np.float32)

    # Drop any NaN groups (just in case)
    valid = ~np.isnan(centroid_hd).any(axis=1)
    groups = [g for g, v in zip(groups, valid) if v]
    centroid_hd = centroid_hd[valid]

    # Center by global mean
    mu_global = features_hd.mean(axis=0).astype(np.float32)
    X = (centroid_hd - mu_global[None, :]).astype(np.float32)

    # t-SNE with safe perplexity for tiny G
    G = X.shape[0]
    perplexity = max(1, min(30, G - 1))  # ensures perplexity < G; e.g., G=2 -> 1
    Z = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        n_iter=1000,
        n_iter_without_progress=250,
    ).fit_transform(X).astype(np.float32)

    return {
        'centroid_group_names': groups,   # [g1, g2, ...]
        'centroid_tsne2d': Z,             # (G, 2)
        'mu_global_hd': mu_global,        # (D,)  (kept for provenance)
    }


def cosine_intra_class(latent_space, labels):
    """
    Computes pairwise cosine similarity within each class.
    Returns: dict[label] -> list of similarities
    """
    scores = defaultdict(list)
    for label in np.unique(labels):
        vecs = latent_space[labels == label]
        if len(vecs) < 2:
            continue
        sim_matrix = cosine_similarity(vecs)
        n = sim_matrix.shape[0]
        pairwise_sims = sim_matrix[np.triu_indices(n, k=1)]
        scores[label] = pairwise_sims
    return scores


def cosine_inter_class(latent_space, labels):
    """
    Computes cosine similarity between class centroids.
    Returns: matrix, class label order
    """
    centroids = []
    class_order = sorted(np.unique(labels))
    for label in class_order:
        centroids.append(latent_space[labels == label].mean(axis=0))
    centroids = np.stack(centroids)
    return cosine_similarity(centroids), class_order


def euclidean_intra_class(latent_space, labels):
    """
    Computes pairwise Euclidean distances within each class.
    Returns: dict[label] -> list of distances
    """
    scores = defaultdict(list)
    for label in np.unique(labels):
        vecs = latent_space[labels == label]
        if len(vecs) < 2:
            continue
        dists = pdist(vecs, metric='euclidean')
        scores[label] = dists
    return scores


def euclidean_inter_class(latent_space, labels):
    """
    Computes Euclidean distances between class centroids.
    Returns: matrix, class label order
    """
    centroids = []
    class_order = sorted(np.unique(labels))
    for label in class_order:
        centroids.append(latent_space[labels == label].mean(axis=0))
    centroids = np.stack(centroids)
    dist_matrix = squareform(pdist(centroids, metric='euclidean'))
    return dist_matrix, class_order


def silhouette_scores(latent_space, labels):
    """
    Computes silhouette score per sample.
    Returns: dict[label] -> list of silhouette scores
    """
    raw_scores = silhouette_samples(latent_space, labels)
    scores = defaultdict(list)
    for score, label in zip(raw_scores, labels):
        scores[label].append(score)
    return scores

