"""
Diagnostic: Per-class accuracy drop under neuropil knockouts
=============================================================
For each of the N trained C16/E16/H16 model runs:
  1. Baseline forward pass on the normal test set → per-class accuracy
  2. 12 neuropil KO forward passes → per-class accuracy each
  3. ΔAccuracy = baseline − KO per class

Aggregates mean and std of ΔAccuracy across all runs.
Saves a CSV and a heatmap (16 classes × 12 neuropils, values = mean ΔAcc).

Usage (from repo root):
    python scripts/diagnostics/diag_KO_permutation.py

Outputs:
    results/diagnostics/KO_permutation/KO_delta_accuracy_mean.csv
    results/diagnostics/KO_permutation/KO_delta_accuracy_std.csv
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

TASK        = 'State_Modality_Valence_16'
BASE_RUNS   = os.path.join('results', 'chkpt_runs')
RUN_PREFIX  = f'{TASK}_C16_E16_H16_'       # X = 1..50
N_RUNS      = 50

NEUROPILS = ['AL', 'MB', 'PENP', 'VLNP', 'CX', 'GNG',
             'LX', 'SNP', 'INP', 'LH', 'OL', 'VMNP']

OUT_DIAG = 'results/diagnostics/KO_permutation'
OUT_PLOT = 'results/CombiPlots'
BATCH_SIZE  = 256
NUM_WORKERS = 4


# ── helpers ───────────────────────────────────────────────────────────────────

def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Row-normalised confusion matrix diagonal → (n_classes,) accuracy array."""
    counts = np.zeros((n_classes, n_classes), dtype=np.int64)
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

    baseline: {allTs_base}/meanZ_allTs
    KO:       {allTs_base}/meanZ_allTs_KO_{neuropil}
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

    # ── enumerate run directories ─────────────────────────────────────────────
    run_dirs = [Path(BASE_RUNS) / f'{RUN_PREFIX}{x}' for x in range(1, N_RUNS + 1)]
    run_dirs = [d for d in run_dirs if d.exists()]
    if not run_dirs:
        raise FileNotFoundError(f'No run directories found under {BASE_RUNS} matching {RUN_PREFIX}*')
    logger.info(f'Found {len(run_dirs)} run directories')

    # ── load base config (shared across all runs) ─────────────────────────────
    base_config  = load_config()
    model_params = base_config['model']['parameters']

    device       = base_config['device']

    # Restore training config to get class names and allTs_path
    with open(f"{base_config['paths']['results_root']}/config.pkl", 'rb') as fh:
        train_config = pickle.load(fh)

    class_names = train_config['data']['classes']
    n_classes   = len(class_names)
    allTs_base  = train_config['paths']['allTs_path']

    logger.info(f'Task: {TASK} | {n_classes} classes | {len(NEUROPILS)} neuropils')
    logger.info(f'allTs_path (baseline): {allTs_base}')

    # ── load test split (same for all runs — split is fixed by task) ──────────
    with open(base_config['paths']['pickle_path'], 'rb') as fh:
        _, _, X_test, _, _, Y_test = pickle.load(fh)
    logger.info(f'Test samples (raw): {len(X_test)}')

    # Pre-build KO test path lists (path rewriting is run-independent)
    X_test_ko = {}
    for neuropil in NEUROPILS:
        cfg_ko = ko_config(train_config, neuropil)
        X_test_ko[neuropil] = paths2neuropilpaths(list(X_test), cfg_ko)

    # Warn once about missing KO directories
    missing = [n for n in NEUROPILS
               if not Path(ko_allTs_path(allTs_base, n)).exists()]
    if missing:
        logger.warning(f'KO data directories not found, will skip: {missing}')

    # ── outer loop: one model per run ─────────────────────────────────────────
    all_deltas = []   # list of (n_classes, n_neuropils) arrays, one per run

    for run_dir in run_dirs:
        models_dir = str(run_dir / 'models' / 'best') + '/'

        if not Path(models_dir).exists():
            logger.warning(f'models/best not found in {run_dir}, skipping.')
            continue

        logger.info(f'── Run {run_dir.name} ──')

        # Load model
        classifier, *_ = load_model(CNN_Transformer, model_params,
                                     models_dir, device, logger)
        if classifier is None:
            logger.warning(f'  Checkpoint missing in {run_dir.name}, skipping.')
            continue
        classifier.eval()

        # Baseline
        loader_base      = make_loader(X_test, Y_test, model_params, allTs_base)
        y_pred_base, y_true = get_predictions(classifier, loader_base, device)
        acc_base         = per_class_accuracy(y_true, y_pred_base, n_classes)
        logger.info(f'  Baseline mean acc: {np.nanmean(acc_base):.3f}')

        # 12 KO runs
        delta_i = np.full((n_classes, len(NEUROPILS)), np.nan)
        for j, neuropil in enumerate(NEUROPILS):
            if neuropil in missing:
                continue
            allTs_ko  = ko_allTs_path(allTs_base, neuropil)
            loader_ko = make_loader(X_test_ko[neuropil], Y_test, model_params, allTs_ko)
            y_pred_ko, _ = get_predictions(classifier, loader_ko, device)
            acc_ko        = per_class_accuracy(y_true, y_pred_ko, n_classes)
            delta_i[:, j] = acc_base - acc_ko

        all_deltas.append(delta_i)
        del classifier
        torch.cuda.empty_cache()

    if not all_deltas:
        raise RuntimeError('No runs completed successfully.')

    logger.info(f'Completed {len(all_deltas)} / {len(run_pkls)} runs.')

    # ── aggregate ─────────────────────────────────────────────────────────────
    stack      = np.stack(all_deltas, axis=0)   # (n_runs, n_classes, n_neuropils)
    delta_mean = np.nanmean(stack, axis=0) * 100  # percentage points
    delta_std  = np.nanstd(stack,  axis=0) * 100

    # ── save CSVs ─────────────────────────────────────────────────────────────
    df_mean = pd.DataFrame(delta_mean, index=class_names, columns=NEUROPILS)
    df_std  = pd.DataFrame(delta_std,  index=class_names, columns=NEUROPILS)
    df_mean.to_csv(os.path.join(OUT_DIAG, 'KO_delta_accuracy_mean.csv'))
    df_std.to_csv( os.path.join(OUT_DIAG, 'KO_delta_accuracy_std.csv'))
    logger.info(f'Saved CSVs → {OUT_DIAG}')

    # ── heatmap (mean ΔAccuracy) ───────────────────────────────────────────────
    vmax = np.nanpercentile(np.abs(delta_mean), 95)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        df_mean,
        ax=ax,
        cmap='RdBu_r',
        center=0,
        vmin=-vmax, vmax=vmax,
        annot=True, fmt='.1f',
        linewidths=0.3,
        cbar_kws={'label': 'Mean ΔAccuracy  (baseline − KO)  [pp]'},
    )
    ax.set_title(
        f'Per-class accuracy drop under neuropil knockouts\n'
        f'({TASK}, n={len(all_deltas)} runs)',
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
