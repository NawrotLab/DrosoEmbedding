"""
Shared neuropil mask utilities for preprocessing and diagnostics.
"""

import json
import os
import pickle
import random
from pathlib import Path

import nibabel as nib
import numpy as np

MASK_DIR       = '/projects/lab-data/Collaboration/Gruenwald_Kadow/Neuropils12_Masks'
SIZES_CACHE    = 'results/preprocessing/neuropil_sizes.json'
SIZES_2D_CACHE = 'results/preprocessing/neuropil_sizes_2d.json'

NEUROPIL_NAMES = np.array(['AL', 'MB', 'PENP', 'VLNP', 'CX', 'GNG',
                            'LX', 'SNP', 'INP', 'LH', 'OL', 'VMNP'])
N_SIZE_SAMPLES = 25


def mask_path(rec_nr: str) -> str:
    return os.path.join(MASK_DIR, f'Neuropils12Registered_{rec_nr}.nii')


def load_mask_3d(rec_nr: str, neuropil_idx: int):
    """Return boolean (y, x, z) mask for the given neuropil, or None if file missing."""
    path = mask_path(rec_nr)
    if not os.path.exists(path):
        return None
    raw = nib.load(path).get_fdata()[:, :, :, neuropil_idx]  # (dim0, dim1, z)
    return (np.transpose(raw, (1, 0, 2)) > 0)                # (y, x, z) bool


def footprint_2d(mask_3d: np.ndarray) -> np.ndarray:
    """Project 3D mask to 2D: True where any Z slice is masked."""
    return np.any(mask_3d, axis=2)  # (y, x) bool


def compute_neuropil_sizes(config: dict, n_samples: int = N_SIZE_SAMPLES,
                            rng: random.Random = None, logger=None) -> np.ndarray:
    """
    Return (12,) array of mean 3D voxel counts per neuropil averaged over
    n_samples training-set recordings. Results cached in SIZES_CACHE.
    """
    if rng is None:
        rng = random.Random(42)

    os.makedirs(os.path.dirname(SIZES_CACHE), exist_ok=True)

    if os.path.exists(SIZES_CACHE):
        with open(SIZES_CACHE) as f:
            cached = json.load(f)
        sizes = np.array([cached[n] for n in NEUROPIL_NAMES], dtype=np.float64)
        if logger:
            logger.info('Loaded cached neuropil 3D voxel sizes:')
            for name, size in zip(NEUROPIL_NAMES, sizes):
                logger.info(f'  {name:<6}: {int(size):,} voxels')
        return sizes

    if logger:
        logger.info(f'Computing neuropil 3D voxel sizes from {n_samples} recordings...')

    with open(config['paths']['pickle_path'], 'rb') as f:
        X_train = pickle.load(f)[0]

    rec_names = list({Path(str(p)).parent.name for p in X_train})
    sample    = rng.sample(rec_names, min(n_samples, len(rec_names)))

    counts = []
    for rec_name in sample:
        rec_nr = rec_name.split('_')[-1]
        mp     = mask_path(rec_nr)
        if not os.path.exists(mp):
            continue
        mask_data    = nib.load(mp).get_fdata()                          # (dim0, dim1, z, 12)
        voxel_counts = np.array([(mask_data[:, :, :, i] > 0).sum()
                                  for i in range(len(NEUROPIL_NAMES))], dtype=np.float64)
        counts.append(voxel_counts)

    if not counts:
        if logger:
            logger.warning('No mask files found — returning uniform sizes of 1.')
        return np.ones(len(NEUROPIL_NAMES), dtype=np.float64)

    sizes = np.mean(counts, axis=0)

    if logger:
        logger.info('Neuropil 3D voxel sizes (mean over sample):')
        for name, size in zip(NEUROPIL_NAMES, sizes):
            logger.info(f'  {name:<6}: {int(size):,} voxels')

    cache = {n: float(sizes[i]) for i, n in enumerate(NEUROPIL_NAMES)}
    with open(SIZES_CACHE, 'w') as f:
        json.dump(cache, f, indent=2)

    return sizes


def compute_neuropil_sizes_2d(config: dict, n_samples: int = N_SIZE_SAMPLES,
                               rng: random.Random = None, logger=None) -> np.ndarray:
    """
    Return (12,) array of mean 2D pixel counts per neuropil (Z-projection footprint)
    averaged over n_samples training-set recordings. Results cached in SIZES_2D_CACHE.

    The 2D footprint counts pixels where any Z-slice is masked — matching the meanZ
    projection used as model input.
    """
    if rng is None:
        rng = random.Random(42)

    os.makedirs(os.path.dirname(SIZES_2D_CACHE), exist_ok=True)

    if os.path.exists(SIZES_2D_CACHE):
        with open(SIZES_2D_CACHE) as f:
            cached = json.load(f)
        sizes = np.array([cached[n] for n in NEUROPIL_NAMES], dtype=np.float64)
        if logger:
            logger.info('Loaded cached neuropil 2D footprint sizes:')
            for name, size in zip(NEUROPIL_NAMES, sizes):
                logger.info(f'  {name:<6}: {int(size):,} pixels')
        return sizes

    if logger:
        logger.info(f'Computing neuropil 2D footprint sizes from {n_samples} recordings...')

    with open(config['paths']['pickle_path'], 'rb') as f:
        X_train = pickle.load(f)[0]

    rec_names = list({Path(str(p)).parent.name for p in X_train})
    sample    = rng.sample(rec_names, min(n_samples, len(rec_names)))

    counts = []
    for rec_name in sample:
        rec_nr = rec_name.split('_')[-1]
        mp     = mask_path(rec_nr)
        if not os.path.exists(mp):
            continue
        mask_data    = nib.load(mp).get_fdata()   # (dim0, dim1, z, 12)
        pixel_counts = np.array([
            np.any(mask_data[:, :, :, i] > 0, axis=2).sum()
            for i in range(len(NEUROPIL_NAMES))
        ], dtype=np.float64)
        counts.append(pixel_counts)

    if not counts:
        if logger:
            logger.warning('No mask files found — returning uniform sizes of 1.')
        return np.ones(len(NEUROPIL_NAMES), dtype=np.float64)

    sizes = np.mean(counts, axis=0)

    if logger:
        logger.info('Neuropil 2D footprint sizes (mean over sample):')
        for name, size in zip(NEUROPIL_NAMES, sizes):
            logger.info(f'  {name:<6}: {int(size):,} pixels')

    cache = {n: float(sizes[i]) for i, n in enumerate(NEUROPIL_NAMES)}
    with open(SIZES_2D_CACHE, 'w') as f:
        json.dump(cache, f, indent=2)

    return sizes
