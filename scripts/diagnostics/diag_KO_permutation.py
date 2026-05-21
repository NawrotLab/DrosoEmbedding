"""
Diagnostic: Per-class accuracy drop under neuropil knockouts
=============================================================
Runs the best C16 model on the normal test set (baseline), then on each of
the 12 neuropil KO datasets.  Computes ΔAccuracy = baseline − KO per class,
saves a CSV and a heatmap (16 classes × 12 neuropils).

Usage (from repo root):
    python scripts/diagnostics/diag_KO_permutation.py

Outputs:
    results/diagnostics/KO_permutation/KO_delta_accuracy.csv
    results/CombiPlots/pdfs/diag_KO_permutation.pdf
    results/CombiPlots/pngs/diag_KO_permutation.png
"""

import copy
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.dataset import CustomDataset
from src.models.cnn_transformer import CNN_Transformer
from src.models.model_io import load_model
from src.utils.config_loader import load_config
from src.utils.helpers import paths2neuropilpaths, get_predictions
from src.utils.logger import setup_logger
from src.visualization.figure_base import apply_style, FONT_SIZES, save_figure

apply_style()

# ── constants ─────────────────────────────────────────────────────────────────

NEUROPILS = ['AL', 'MB', 'PENP', 'VLNP', 'CX', 'GNG',
             'LX', 'SNP', 'INP', 'LH', 'OL', 'VMNP']

OUT_DIAG = 'results/diagnostics/KO_permutation'
OUT_PLOT = 'results/CombiPlots'
BATCH_SIZE  = 256
NUM_WORKERS = 4


# ── helpers ───────────────────────────────────────────────────────────────────

def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Row-normalised confusion matrix diagonal → (n_classes,) accuracy array."""
    counts  = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        counts[t, p] += 1
    row_sums = counts.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(row_sums > 0, counts.diagonal() / row_sums, np.nan)


def make_loader(X, Y, model_params, allTs_path):
    ds = CustomDataset(
        X, Y,
        transform=True,
        seq_length=model_params['seq_len'],
        seq_steps=model_params['seq_steps'],
        allTs_path=allTs_path,
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


def ko_allTs_path(base_allTs_path: str, neuropil: str) -> str:
    """Derive KO allTs path from baseline.

    baseline:  {allTs_base}/meanZ_allTs
    KO:        {allTs_base}/meanZ_allTs_KO_{neuropil}
    """
    parent = os.path.dirname(base_allTs_path.rstrip('/'))
    return os.path.join(parent, f'meanZ_allTs_KO_{neuropil}')


def ko_config(base_cfg: dict, neuropil: str) -> dict:
    """Deep-copy base config with remove_neuropil=True for the given neuropil."""
    cfg = copy.deepcopy(base_cfg)
    cfg['data']['preprocessing']['remove_neuropil']  = True
    cfg['data']['preprocessing']['isolate_neuropil'] = False
    cfg['data']['preprocessing']['neuropil']         = neuropil
    cfg['paths']['allTs_path'] = ko_allTs_path(base_cfg['paths']['allTs_path'], neuropil)
    return cfg


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIAG, exist_ok=True)
    os.makedirs(OUT_PLOT, exist_ok=True)

    logger = setup_logger(task_name='diag_KO_permutation',
                          log_dir='logs/diag_KO_permutation')

    # ── config & paths ────────────────────────────────────────────────────────
    config = load_config()
    paths        = config['paths']
    model_params = config['model']['parameters']

    # Restore training config (contains derived paths and class names)
    with open(f"{paths['results_root']}/config.pkl", 'rb') as fh:
        config = pickle.load(fh)

    device      = config['device']
    class_names = config['data']['classes']
    n_classes   = len(class_names)
    allTs_base  = config['paths']['allTs_path']

    logger.info(f"Task: {config['data']['task']}, {n_classes} classes")
    logger.info(f"allTs_path (baseline): {allTs_base}")

    # ── test split ────────────────────────────────────────────────────────────
    with open(paths['pickle_path'], 'rb') as fh:
        _, _, X_test, _, _, Y_test = pickle.load(fh)
    logger.info(f"Test samples (raw): {len(X_test)}")

    # ── load model once ───────────────────────────────────────────────────────
    classifier, *_ = load_model(CNN_Transformer, model_params,
                                paths['models'], device, logger)
    if classifier is None:
        raise FileNotFoundError('Trained model checkpoint not found.')
    classifier.eval()

    # ── baseline ──────────────────────────────────────────────────────────────
    logger.info('Running baseline ...')
    loader_base  = make_loader(X_test, Y_test, model_params, allTs_base)
    y_pred_base, y_true = get_predictions(classifier, loader_base, device)
    acc_base = per_class_accuracy(y_true, y_pred_base, n_classes)
    logger.info(f'Baseline mean accuracy: {np.nanmean(acc_base):.3f}')

    # ── per-neuropil KO ───────────────────────────────────────────────────────
    delta = np.full((n_classes, len(NEUROPILS)), np.nan)  # rows=classes, cols=neuropils

    for j, neuropil in enumerate(NEUROPILS):
        logger.info(f'KO: {neuropil} ...')

        cfg_ko       = ko_config(config, neuropil)
        allTs_ko     = cfg_ko['paths']['allTs_path']
        X_test_ko    = paths2neuropilpaths(list(X_test), cfg_ko)

        if not Path(allTs_ko).exists():
            logger.warning(f'KO data directory not found: {allTs_ko} — skipping.')
            continue

        loader_ko        = make_loader(X_test_ko, Y_test, model_params, allTs_ko)
        y_pred_ko, _     = get_predictions(classifier, loader_ko, device)
        acc_ko           = per_class_accuracy(y_true, y_pred_ko, n_classes)
        delta[:, j]      = acc_base - acc_ko
        logger.info(f'  Mean ΔAcc: {np.nanmean(delta[:, j]):.4f}')

    # ── save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(delta * 100,          # convert to percentage points
                      index=class_names,
                      columns=NEUROPILS)
    csv_path = os.path.join(OUT_DIAG, 'KO_delta_accuracy.csv')
    df.to_csv(csv_path)
    logger.info(f'Saved CSV → {csv_path}')

    # ── heatmap ───────────────────────────────────────────────────────────────
    vmax = np.nanpercentile(np.abs(delta * 100), 95)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        df,
        ax=ax,
        cmap='RdBu_r',
        center=0,
        vmin=-vmax, vmax=vmax,
        annot=True, fmt='.1f',
        linewidths=0.3,
        cbar_kws={'label': 'ΔAccuracy  (baseline − KO)  [pp]'},
    )
    ax.set_title(
        'Per-class accuracy drop under neuropil knockouts\n'
        f'({config["data"]["task"]})',
        fontsize=FONT_SIZES['title'],
    )
    ax.set_xlabel('Neuropil knocked out', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('Class',                fontsize=FONT_SIZES['label'])
    ax.tick_params(axis='both', labelsize=FONT_SIZES['tick'])
    plt.tight_layout()

    save_figure(fig, os.path.join(OUT_PLOT, 'diag_KO_permutation.pdf'),
                formats=('pdf', 'png'))
    logger.info('Done.')


if __name__ == '__main__':
    main()
