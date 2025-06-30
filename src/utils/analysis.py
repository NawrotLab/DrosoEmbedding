import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


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

