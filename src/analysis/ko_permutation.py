"""
KO permutation analysis: core computation and aggregation functions.

The heavy computation (50 runs × 12 neuropils forward passes) is isolated in
run_ko_permutation; load_or_run_ko_permutation wraps it with disk caching so
subsequent runs skip straight to aggregation and plotting.
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CustomDataset
from src.models.cnn_transformer import CNN_Transformer
from src.models.model_io import load_model
from src.utils.helpers import get_predictions_with_probs
from src.utils.neuropil_masks import compute_neuropil_sizes_2d

# ── constants ─────────────────────────────────────────────────────────────────

NEUROPILS = ['AL', 'MB', 'PENP', 'VLNP', 'CX', 'GNG',
             'LX', 'SNP', 'INP', 'LH', 'OL', 'VMNP']

GROUPS = {
    'Odor':       [0, 1, 2, 3],
    'Taste':      [4, 5, 6, 7],
    'Combined':   [8, 9, 10, 11, 12, 13, 14, 15],
    'Appetitive': [0, 2, 4, 6, 8, 12],
    'Aversive':   [1, 3, 5, 7, 9, 13],
    'Conflict':   [10, 11, 14, 15],
    'Starved':    [0, 1, 4, 5, 8, 9, 10, 11],
    'Fed':        [2, 3, 6, 7, 12, 13, 14, 15],
}
GROUP_ORDER = list(GROUPS.keys())

CONTRAST_PAIRS = {
    'Starved_minus_Fed':         ('Starved',    'Fed'),
    'Odor_minus_Taste':          ('Odor',       'Taste'),
    'Appetitive_minus_Aversive': ('Appetitive', 'Aversive'),
}


# ── low-level helpers ─────────────────────────────────────────────────────────

def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                        n_classes: int) -> np.ndarray:
    """Diagonal of the row-normalised confusion matrix → (n_classes,)."""
    counts = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        counts[t, p] += 1
    row_sums = counts.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(row_sums > 0, counts.diagonal() / row_sums, np.nan)


def ko_allTs_path(base_allTs_path: str, neuropil: str, variant: str = '') -> str:
    """Derive the KO data directory path for a given neuropil and fill variant."""
    parent = os.path.dirname(base_allTs_path.rstrip('/'))
    suffix = f'_{variant}' if variant in ('noisefill', 'static', 'shuffled') else ''
    return os.path.join(parent, f'meanZ_allTs_KO{suffix}_{neuropil}')


def make_ko_loader(X, Y, model_params, allTs_path,
                   batch_size: int = 256, num_workers: int = 4) -> DataLoader:
    """Build a DataLoader pointing at allTs_path as the frame root directory."""
    ds = CustomDataset(
        X, Y,
        transform=True,
        seq_length=model_params['seq_len'],
        seq_steps=model_params['seq_steps'],
        allTs_path=allTs_path,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def find_common_valid_entries(X_test, Y_test, model_params,
                               allTs_base, available_ko_dirs):
    """Keep only test sequences whose frames exist in baseline AND every KO dir.

    Guarantees that baseline and all KO loaders produce the same samples in the
    same order — a prerequisite for aligned per-class accuracy comparisons.
    """
    seq_len   = model_params['seq_len']
    seq_steps = model_params['seq_steps']

    valid_X, valid_Y = [], []
    for path, label in zip(X_test, Y_test):
        path = str(path)
        recording = path.split('/')[-2]
        try:
            start = int(path.split('_')[-1].split('.')[0])
        except ValueError:
            continue

        frames = list(range(start, start + (seq_len - 1) * seq_steps + 1, seq_steps))

        if not all(
            (Path(allTs_base) / recording / f'{recording}_{f}.tiff').exists()
            for f in frames
        ):
            continue

        ok = True
        for ko_dir in available_ko_dirs:
            if not all(
                (Path(ko_dir) / recording / f'{recording}_{f}.tiff').exists()
                for f in frames
            ):
                ok = False
                break

        if ok:
            valid_X.append(path)
            valid_Y.append(label)

    return valid_X, valid_Y


# ── core computation ──────────────────────────────────────────────────────────

def run_ko_permutation(
    run_dirs, X_test, Y_test, model_params, allTs_base,
    neuropils, variant, device, n_classes,
    batch_size: int = 256, num_workers: int = 4,
    logger=None,
) -> np.ndarray:
    """Run baseline + KO forward passes across all model runs.

    Parameters
    ----------
    run_dirs   : list[Path]  — one per trained model (e.g. 50 runs)
    variant    : str         — KO fill variant ('static', 'shuffled', …)

    Returns
    -------
    stack : ndarray (n_completed_runs, n_classes, n_neuropils)
        (baseline_acc − KO_acc) in percentage points; NaN where KO dir missing.
    """
    def _log(msg):
        if logger: logger.info(msg)
        else: print(msg)

    missing = [n for n in neuropils
               if not Path(ko_allTs_path(allTs_base, n, variant)).exists()]
    if missing:
        _log(f'KO dirs missing (will be NaN): {missing}')

    available_ko_dirs = [ko_allTs_path(allTs_base, n, variant)
                         for n in neuropils if n not in missing]

    _log('Pre-filtering test set to sequences present in baseline + all KO dirs…')
    X_filt, Y_filt = find_common_valid_entries(
        list(X_test), list(Y_test), model_params, allTs_base, available_ko_dirs,
    )
    _log(f'Test samples after filtering: {len(X_filt)}')
    if not X_filt:
        raise RuntimeError('No valid test samples after intersection filter.')

    all_deltas = []

    for run_dir in run_dirs:
        models_dir = str(run_dir / 'models' / 'best') + '/'
        if not Path(models_dir).exists():
            _log(f'models/best not found in {run_dir.name}, skipping.')
            continue

        _log(f'── Run {run_dir.name} ──')
        classifier, *_ = load_model(CNN_Transformer, model_params,
                                     models_dir, device, logger=logger)
        if classifier is None:
            _log(f'Checkpoint missing in {run_dir.name}, skipping.')
            continue
        classifier.eval()

        loader_base = make_ko_loader(X_filt, Y_filt, model_params, allTs_base,
                                      batch_size, num_workers)
        y_pred_base, y_true, _ = get_predictions_with_probs(
            classifier, loader_base, device)
        acc_base = per_class_accuracy(y_true, y_pred_base, n_classes)
        _log(f'  Baseline mean acc: {np.nanmean(acc_base):.3f}')

        delta_i = np.full((n_classes, len(neuropils)), np.nan)
        for j, neuropil in enumerate(neuropils):
            if neuropil in missing:
                continue
            allTs_ko  = ko_allTs_path(allTs_base, neuropil, variant)
            loader_ko = make_ko_loader(X_filt, Y_filt, model_params, allTs_ko,
                                        batch_size, num_workers)
            y_pred_ko, _, _ = get_predictions_with_probs(
                classifier, loader_ko, device)
            acc_ko        = per_class_accuracy(y_true, y_pred_ko, n_classes)
            delta_i[:, j] = (acc_base - acc_ko) * 100   # percentage points
            _log(f'  KO {neuropil:<6}  Δacc = {np.nanmean(delta_i[:, j]):+.2f} pp')

        all_deltas.append(delta_i)
        del classifier
        torch.cuda.empty_cache()

    if not all_deltas:
        raise RuntimeError('No runs completed successfully.')

    _log(f'Completed {len(all_deltas)}/{len(run_dirs)} runs.')
    return np.stack(all_deltas, axis=0)   # (n_runs, n_classes, n_neuropils)


# ── caching wrapper ───────────────────────────────────────────────────────────

def load_or_run_ko_permutation(
    cache_path: str,
    recompute: bool = False,
    logger=None,
    **kwargs,
) -> np.ndarray:
    """Load cached KO stack, or run the full permutation and save the result.

    The cache is a single .npy file containing the raw
    (n_runs, n_classes, n_neuropils) delta-accuracy stack.
    Pass recompute=True to ignore the cache and rerun everything.
    All extra kwargs are forwarded to run_ko_permutation.
    """
    def _log(msg):
        if logger: logger.info(msg)
        else: print(msg)

    if not recompute and os.path.exists(cache_path):
        _log(f'Loading cached KO stack ({cache_path})')
        return np.load(cache_path)

    _log('Cache not found or recompute=True — running KO permutation (slow)…')
    stack = run_ko_permutation(logger=logger, **kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.save(cache_path, stack)
    _log(f'Saved KO stack → {cache_path}')
    return stack


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate_ko_by_group(
    stack: np.ndarray,
    neuropils,
    config: dict = None,
    rng: random.Random = None,
    logger=None,
    groups: dict = None,
    group_order: list = None,
    contrast_pairs: dict = None,
):
    """Aggregate the raw KO stack into group profiles and contrasts.

    Parameters
    ----------
    stack     : (n_runs, n_classes, n_neuropils) in pp — from load_or_run_ko_permutation
    config    : base YAML config dict, needed for the 2D size normalisation

    Returns
    -------
    df_group    : DataFrame (n_groups × n_neuropils)
    df_contrast : DataFrame (n_contrasts × n_neuropils)
    """
    if groups is None:
        groups = GROUPS
    if group_order is None:
        group_order = GROUP_ORDER
    if contrast_pairs is None:
        contrast_pairs = CONTRAST_PAIRS

    delta_mean = np.nanmean(stack, axis=0)   # (n_classes, n_neuropils)

    # Normalise by each neuropil's mean 2D pixel footprint (per 1k pixels)
    sizes = compute_neuropil_sizes_2d(config, rng=rng, logger=logger)
    delta_mean = delta_mean / (sizes / 1_000)

    n_classes = delta_mean.shape[0]

    group_profiles = {}
    for gname, cls_idx in groups.items():
        sel = [i for i in cls_idx if i < n_classes]
        group_profiles[gname] = np.nanmean(delta_mean[sel, :], axis=0)

    df_group = pd.DataFrame(group_profiles, index=neuropils).T
    df_group = df_group.loc[group_order]

    contrast_rows = {}
    for cname, (g1, g2) in contrast_pairs.items():
        if g1 in group_profiles and g2 in group_profiles:
            contrast_rows[cname] = group_profiles[g1] - group_profiles[g2]
    df_contrast = pd.DataFrame(contrast_rows, index=neuropils).T

    return df_group, df_contrast
