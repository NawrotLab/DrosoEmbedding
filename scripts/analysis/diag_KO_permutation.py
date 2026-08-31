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
    python -m scripts.analysis.diag_KO_permutation

Outputs:
    results/diagnostics/KO_permutation/KO_delta_accuracy_mean.csv
    results/diagnostics/KO_permutation/KO_delta_accuracy_std.csv
    results/CombiPlots/pdfs/diag_KO_permutation.pdf
    results/CombiPlots/pngs/diag_KO_permutation.png
"""

import argparse
import copy
import os
import pickle
import random
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
from src.utils.helpers import paths2neuropilpaths, get_predictions_with_probs
from src.utils.logger import setup_logger
from src.utils.neuropil_masks import compute_neuropil_sizes, compute_neuropil_sizes_2d
from src.visualization.figure_base import apply_style, FONT_SIZES, save_figure
from src.visualization.visualize_interpretability import plot_contrasts_horizontal

apply_style()

# ── constants ─────────────────────────────────────────────────────────────────

TASK        = 'State_Modality_Valence_16'
BASE_RUNS   = os.path.join('results', 'chkpt_runs')
RUN_PREFIX  = f'{TASK}_C16_E16_H16_'       # X = 1..50
N_RUNS      = 50

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

OUT_DIAG = 'results/diagnostics/KO_permutation'
OUT_PLOT = 'results/CombiPlots'
BATCH_SIZE  = 256
NUM_WORKERS = 4


# ── helpers ───────────────────────────────────────────────────────────────────

def prediction_entropy(probs: np.ndarray) -> float:
    """Mean Shannon entropy (nats) over samples. probs: (N, n_classes)."""
    eps = 1e-10
    return float(-np.sum(probs * np.log(probs + eps), axis=1).mean())


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


def ko_allTs_path(base_allTs_path: str, neuropil: str, variant: str = '') -> str:
    """Derive KO allTs path from baseline.

    baseline: {allTs_base}/meanZ_allTs
    KO:       {allTs_base}/meanZ_allTs_KO_{neuropil}
    noisefill: {allTs_base}/meanZ_allTs_KO_noisefill_{neuropil}
    """
    parent = os.path.dirname(base_allTs_path.rstrip('/'))
    suffix = f'_{variant}' if variant in ('noisefill', 'static', 'shuffled') else ''
    return os.path.join(parent, f'meanZ_allTs_KO{suffix}_{neuropil}')


def ko_config(base_cfg: dict, neuropil: str, variant: str = '') -> dict:
    """Deep-copy base config with remove_neuropil=True for the given neuropil."""
    cfg = copy.deepcopy(base_cfg)
    cfg['data']['preprocessing']['remove_neuropil']  = True
    cfg['data']['preprocessing']['isolate_neuropil'] = False
    cfg['data']['preprocessing']['neuropil']         = neuropil
    cfg['paths']['allTs_path'] = ko_allTs_path(base_cfg['paths']['allTs_path'], neuropil, variant)
    return cfg


def find_common_valid_entries(
    X_test: list,
    Y_test: list,
    model_params: dict,
    allTs_base: str,
    available_ko_dirs: list,
) -> tuple:
    """Filter (X_test, Y_test) to sequences whose frames exist in baseline + every KO dir.

    Replicates CustomDataset._get_sequence_paths logic so the loaders built from
    the returned lists will never silently drop samples, guaranteeing that
    baseline and all KO predictions are aligned.
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

        # check baseline
        if not all(
            (Path(allTs_base) / recording / f'{recording}_{f}.tiff').exists()
            for f in frames
        ):
            continue

        # check every available KO dir (recording name is the same in each)
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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='',
                        help='KO variant: empty = zero-fill (default), noisefill, static, shuffled')
    args    = parser.parse_args()
    variant = args.variant

    out_diag = OUT_DIAG + (f'_{variant}' if variant else '')
    out_plot = OUT_PLOT
    os.makedirs(out_diag, exist_ok=True)
    os.makedirs(out_plot, exist_ok=True)

    tag = f'_{variant}' if variant else ''
    logger = setup_logger(task_name=f'diag_KO_permutation{tag}',
                          log_dir='logs/diag_KO_permutation')

    # ── enumerate run directories ─────────────────────────────────────────────
    all_expected = [Path(BASE_RUNS) / f'{RUN_PREFIX}{x}' for x in range(1, N_RUNS + 1)]
    missing_dirs = [d.name for d in all_expected if not d.exists()]
    no_ckpt      = [d.name for d in all_expected if d.exists() and not (d / 'models' / 'best').exists()]
    if missing_dirs:
        logger.warning(f'Missing run directories ({len(missing_dirs)}): {missing_dirs}')
    if no_ckpt:
        logger.warning(f'Runs without models/best ({len(no_ckpt)}): {no_ckpt}')
    run_dirs = [d for d in all_expected if d.exists()]
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

    # Identify missing KO directories first
    missing = [n for n in NEUROPILS
               if not Path(ko_allTs_path(allTs_base, n, variant)).exists()]
    if missing:
        logger.warning(f'KO data directories not found, will skip: {missing}')

    available_ko_dirs = [ko_allTs_path(allTs_base, n, variant)
                         for n in NEUROPILS if n not in missing]

    # Pre-filter: keep only sequences whose frames exist in baseline + every KO dir.
    # This guarantees all loaders produce the same number of samples in the same
    # order, so per-class accuracy arrays are aligned across baseline and all KOs.
    # per-class counts before filtering
    raw_X = list(X_test)
    raw_Y = list(Y_test)
    raw_counts = {i: raw_Y.count(i) for i in range(n_classes)}

    logger.info('Pre-filtering test set to sequences present in all KO dirs...')
    X_test, Y_test = find_common_valid_entries(
        list(X_test), list(Y_test), model_params, allTs_base, available_ko_dirs
    )
    logger.info(f'Test samples after filtering: {len(X_test)}')
    if not X_test:
        raise RuntimeError('No valid test samples remain after intersection filter.')

    # report which classes lost all samples
    filtered_counts = {i: list(Y_test).count(i) for i in range(n_classes)}
    dropped_classes = []
    for i, name in enumerate(class_names):
        before, after = raw_counts[i], filtered_counts[i]
        if after == 0 and before > 0:
            logger.warning(f'  Class {i} "{name}": ALL {before} samples dropped by KO filter')
            dropped_classes.append(i)
        elif after < before:
            logger.info(f'  Class {i} "{name}": {before} → {after} samples after filter')

    # ── KO coverage report: per-neuropil missing recordings ──────────────────
    seq_len   = model_params['seq_len']
    seq_steps = model_params['seq_steps']

    # baseline-valid indices (frames exist in baseline dir)
    baseline_valid_idx = []
    for i, path in enumerate(raw_X):
        path = str(path)
        recording = path.split('/')[-2]
        try:
            start = int(path.split('_')[-1].split('.')[0])
        except ValueError:
            continue
        frames = list(range(start, start + (seq_len - 1) * seq_steps + 1, seq_steps))
        if all((Path(allTs_base) / recording / f'{recording}_{f}.tiff').exists() for f in frames):
            baseline_valid_idx.append(i)
    n_baseline = len(baseline_valid_idx)

    logger.info(f'KO preprocessing coverage report ({n_baseline} baseline-valid samples):')
    for neuropil in NEUROPILS:
        ko_dir = ko_allTs_path(allTs_base, neuropil, variant)
        if neuropil in missing:
            logger.warning(f'  {neuropil:<6} — directory missing entirely, needs full preprocessing')
            continue

        dropped_cls  = set()
        missing_recs = set()
        for i in baseline_valid_idx:
            path = str(raw_X[i])
            recording = path.split('/')[-2]
            try:
                start = int(path.split('_')[-1].split('.')[0])
            except ValueError:
                continue
            frames = list(range(start, start + (seq_len - 1) * seq_steps + 1, seq_steps))
            if not all((Path(ko_dir) / recording / f'{recording}_{f}.tiff').exists() for f in frames):
                dropped_cls.add(raw_Y[i])
                missing_recs.add(recording)

        if not missing_recs:
            logger.info(f'  {neuropil:<6} — OK, all baseline recordings present')
        else:
            affected = ', '.join(class_names[c] for c in sorted(dropped_cls))
            logger.warning(
                f'  {neuropil:<6} — {len(missing_recs)} recordings missing '
                f'| classes affected: {affected}'
            )
            out_file = os.path.join(out_diag, f'missing_recordings_KO_{neuropil}.txt')
            with open(out_file, 'w') as fh:
                fh.write('\n'.join(sorted(missing_recs)))
            logger.info(f'           → missing recording list saved to {out_file}')

    # Pre-build KO test path lists on the filtered set
    X_test_ko = {}
    for neuropil in NEUROPILS:
        if neuropil in missing:
            continue
        cfg_ko = ko_config(train_config, neuropil, variant)
        X_test_ko[neuropil] = paths2neuropilpaths(list(X_test), cfg_ko)

    # ── outer loop: one model per run ─────────────────────────────────────────
    all_deltas          = []   # (n_classes, n_neuropils) per run
    all_entropy_deltas  = []   # (n_neuropils,) per run

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
        loader_base                    = make_loader(X_test, Y_test, model_params, allTs_base)
        y_pred_base, y_true, probs_base = get_predictions_with_probs(classifier, loader_base, device)
        acc_base                        = per_class_accuracy(y_true, y_pred_base, n_classes)
        entropy_base                    = prediction_entropy(probs_base)
        logger.info(f'  Baseline mean acc: {np.nanmean(acc_base):.3f}  entropy: {entropy_base:.3f}')

        # 12 KO runs
        delta_i         = np.full((n_classes,    len(NEUROPILS)), np.nan)
        delta_entropy_i = np.full((len(NEUROPILS),),              np.nan)
        for j, neuropil in enumerate(NEUROPILS):
            if neuropil in missing:
                continue
            allTs_ko  = ko_allTs_path(allTs_base, neuropil, variant)
            loader_ko = make_loader(X_test_ko[neuropil], Y_test, model_params, allTs_ko)
            y_pred_ko, _, probs_ko = get_predictions_with_probs(classifier, loader_ko, device)
            acc_ko                  = per_class_accuracy(y_true, y_pred_ko, n_classes)
            entropy_ko              = prediction_entropy(probs_ko)
            delta_i[:, j]           = acc_base - acc_ko
            delta_entropy_i[j]      = entropy_ko - entropy_base
            mean_delta_pp = np.nanmean(delta_i[:, j]) * 100
            logger.info(
                f'  KO {neuropil:<6} acc={np.nanmean(acc_ko):.3f}  '
                f'Δacc={mean_delta_pp:+.2f} pp  '
                f'Δentropy={delta_entropy_i[j]:+.3f}'
            )

        # rank neuropils by mean |Δ| across classes for this run
        mean_abs_delta = np.nanmean(np.abs(delta_i), axis=0) * 100
        ranked = sorted(
            [(NEUROPILS[j], mean_abs_delta[j]) for j in range(len(NEUROPILS)) if not np.isnan(mean_abs_delta[j])],
            key=lambda x: x[1], reverse=True,
        )
        top3 = ', '.join(f'{n} ({v:.2f} pp)' for n, v in ranked[:3])
        logger.info(f'  Top neuropils: {top3}')

        all_deltas.append(delta_i)
        all_entropy_deltas.append(delta_entropy_i)
        del classifier
        torch.cuda.empty_cache()

    if not all_deltas:
        raise RuntimeError('No runs completed successfully.')

    logger.info(f'Completed {len(all_deltas)} / {len(run_dirs)} runs.')

    # aggregate neuropil ranking across all runs
    stack_preview = np.stack(all_deltas, axis=0)
    mean_abs_all  = np.nanmean(np.abs(stack_preview), axis=(0, 1)) * 100
    ranked_all = sorted(
        [(NEUROPILS[j], mean_abs_all[j]) for j in range(len(NEUROPILS)) if not np.isnan(mean_abs_all[j])],
        key=lambda x: x[1], reverse=True,
    )
    logger.info('Neuropil ranking by mean |ΔAcc| across all runs and classes:')
    for rank, (name, val) in enumerate(ranked_all, 1):
        logger.info(f'  {rank:2d}. {name:<6}  {val:.2f} pp')

    # ── aggregate ─────────────────────────────────────────────────────────────
    stack      = np.stack(all_deltas, axis=0)   # (n_runs, n_classes, n_neuropils)
    delta_mean = np.nanmean(stack, axis=0) * 100  # percentage points
    delta_std  = np.nanstd(stack,  axis=0) * 100

    # ── neuropil 3D voxel sizes ────────────────────────────────────────────────
    rng            = random.Random(42)
    neuropil_sizes = compute_neuropil_sizes(base_config, rng=rng, logger=logger)
    SCALE          = 10_000   # express as ΔAcc per 10k voxels
    delta_mean_norm = delta_mean / (neuropil_sizes / SCALE)   # (n_classes, n_neuropils)

    # ── save CSVs ─────────────────────────────────────────────────────────────
    df_mean = pd.DataFrame(delta_mean, index=class_names, columns=NEUROPILS)
    df_std  = pd.DataFrame(delta_std,  index=class_names, columns=NEUROPILS)
    df_mean.to_csv(os.path.join(out_diag, 'KO_delta_accuracy_mean.csv'))
    df_std.to_csv( os.path.join(out_diag, 'KO_delta_accuracy_std.csv'))
    logger.info(f'Saved CSVs → {OUT_DIAG}')

    # ── heatmap (mean ΔAccuracy) ───────────────────────────────────────────────
    vmax = np.nanpercentile(np.abs(delta_mean), 95)

    TITLE_FS = 20
    LABEL_FS = 16
    TICK_FS  = 14
    ANNOT_FS = 12
    CBAR_FS  = 14

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        df_mean,
        ax=ax,
        cmap='RdBu_r',
        center=0,
        vmin=-vmax, vmax=vmax,
        annot=True, fmt='.1f',
        annot_kws={'size': ANNOT_FS},
        linewidths=0.3,
        cbar_kws={'label': 'Mean ΔAccuracy (baseline − KO) [pp]', 'shrink': 0.8},
    )
    ax.set_title(
        f'Per-class accuracy drop under neuropil knockouts  (n = {len(all_deltas)} runs)',
        fontsize=TITLE_FS,
        pad=14,
    )
    ax.set_xlabel('Neuropil knocked out', fontsize=LABEL_FS, labelpad=8)
    ax.set_ylabel('Behavioural class',    fontsize=LABEL_FS, labelpad=8)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=CBAR_FS)
    cbar.ax.yaxis.label.set_size(CBAR_FS)
    plt.tight_layout()

    save_figure(fig, os.path.join(out_plot, f'diag_KO_permutation{tag}.pdf'),
                formats=('pdf', 'png'))

    # ── grouped heatmap (8 groups × 12 neuropils) ─────────────────────────────
    group_delta = np.array([
        np.nanmean(delta_mean[GROUPS[g], :], axis=0)
        for g in GROUP_ORDER
    ])
    df_group = pd.DataFrame(group_delta, index=GROUP_ORDER, columns=NEUROPILS)
    df_group.to_csv(os.path.join(out_diag, 'KO_delta_accuracy_groups.csv'))

    vmax_g = np.nanpercentile(np.abs(group_delta), 95)
    fig_g, ax_g = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        df_group,
        ax=ax_g,
        cmap='RdBu_r',
        center=0,
        vmin=-vmax_g, vmax=vmax_g,
        annot=True, fmt='.1f',
        annot_kws={'size': ANNOT_FS},
        linewidths=0.3,
        cbar_kws={'label': 'Mean ΔAccuracy (baseline − KO) [pp]', 'shrink': 0.8},
    )
    ax_g.set_title(
        f'Group-level accuracy drop under neuropil knockouts  (n = {len(all_deltas)} runs)',
        fontsize=TITLE_FS,
        pad=14,
    )
    ax_g.set_xlabel('Neuropil knocked out', fontsize=LABEL_FS, labelpad=8)
    ax_g.set_ylabel('Condition group',      fontsize=LABEL_FS, labelpad=8)
    ax_g.tick_params(axis='both', labelsize=TICK_FS)
    cbar_g = ax_g.collections[0].colorbar
    cbar_g.ax.tick_params(labelsize=CBAR_FS)
    cbar_g.ax.yaxis.label.set_size(CBAR_FS)
    plt.tight_layout()

    save_figure(fig_g, os.path.join(out_plot, f'diag_KO_permutation_groups{tag}.pdf'),
                formats=('pdf', 'png'))

    # ── OOD diagnostics: entropy ───────────────────────────────────────────────
    entropy_stack      = np.stack(all_entropy_deltas, axis=0)   # (n_runs, n_neuropils)
    mean_entropy_delta = np.nanmean(entropy_stack, axis=0)       # (n_neuropils,)
    std_entropy_delta  = np.nanstd(entropy_stack,  axis=0)

    df_entropy = pd.DataFrame({
        'neuropil':      NEUROPILS,
        'mean_delta_entropy': mean_entropy_delta,
        'std_delta_entropy':  std_entropy_delta,
    })
    df_entropy.to_csv(os.path.join(out_diag, 'KO_entropy_delta.csv'), index=False)

    # mean ΔAcc per neuropil (averaged over classes) for the scatter
    mean_delta_per_neuropil = np.nanmean(delta_mean, axis=0)   # (n_neuropils,)

    fig_ood, axes_ood = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1 — ΔEntropy bar chart
    ax1 = axes_ood[0]
    colors = ['#d62728' if v > 0 else '#1f77b4' for v in mean_entropy_delta]
    ax1.bar(NEUROPILS, mean_entropy_delta,
            yerr=std_entropy_delta, color=colors,
            capsize=4, edgecolor='white', linewidth=0.5)
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.set_xlabel('Neuropil knocked out', fontsize=LABEL_FS)
    ax1.set_ylabel('ΔEntropy (KO − baseline) [nats]', fontsize=LABEL_FS)
    ax1.set_title('Prediction entropy increase under KO\n(proxy for OOD confusion)',
                  fontsize=TITLE_FS, pad=10)
    ax1.tick_params(axis='both', labelsize=TICK_FS)

    # Panel 2 — ΔAcc vs ΔEntropy scatter
    ax2 = axes_ood[1]
    ax2.scatter(mean_delta_per_neuropil, mean_entropy_delta,
                s=80, color='steelblue', edgecolors='white', linewidths=0.5, zorder=3)
    for name, x, y in zip(NEUROPILS, mean_delta_per_neuropil, mean_entropy_delta):
        ax2.annotate(name, (x, y), textcoords='offset points', xytext=(6, 4),
                     fontsize=TICK_FS - 1)
    ax2.axhline(0, color='grey', linewidth=0.6, linestyle='--')
    ax2.axvline(0, color='grey', linewidth=0.6, linestyle='--')
    ax2.set_xlabel('Mean ΔAccuracy (baseline − KO) [pp]', fontsize=LABEL_FS)
    ax2.set_ylabel('Mean ΔEntropy (KO − baseline) [nats]', fontsize=LABEL_FS)
    ax2.set_title('ΔAcc vs ΔEntropy per neuropil\n(upper-right = OOD suspect)',
                  fontsize=TITLE_FS, pad=10)
    ax2.tick_params(axis='both', labelsize=TICK_FS)

    plt.tight_layout()
    save_figure(fig_ood, os.path.join(out_plot, f'diag_KO_ood_entropy{tag}.pdf'),
                formats=('pdf', 'png'))

    # ── size-normalised heatmap (16 classes × 12 neuropils) ───────────────────
    df_norm = pd.DataFrame(delta_mean_norm, index=class_names, columns=NEUROPILS)
    df_norm.to_csv(os.path.join(out_diag, 'KO_delta_accuracy_norm.csv'))

    vmax_n = np.nanpercentile(np.abs(delta_mean_norm), 95)
    fig_n, ax_n = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        df_norm, ax=ax_n, cmap='RdBu_r', center=0,
        vmin=-vmax_n, vmax=vmax_n,
        annot=True, fmt='.2f',
        annot_kws={'size': ANNOT_FS},
        linewidths=0.3,
        cbar_kws={'label': f'Mean ΔAccuracy per {SCALE:,} voxels [pp / 10k vox]', 'shrink': 0.8},
    )
    ax_n.set_title(
        f'Per-class accuracy drop — size-normalised  (n = {len(all_deltas)} runs)',
        fontsize=TITLE_FS, pad=14,
    )
    ax_n.set_xlabel('Neuropil knocked out', fontsize=LABEL_FS, labelpad=8)
    ax_n.set_ylabel('Behavioural class',    fontsize=LABEL_FS, labelpad=8)
    ax_n.tick_params(axis='both', labelsize=TICK_FS)
    cbar_n = ax_n.collections[0].colorbar
    cbar_n.ax.tick_params(labelsize=CBAR_FS)
    cbar_n.ax.yaxis.label.set_size(CBAR_FS)
    plt.tight_layout()
    save_figure(fig_n, os.path.join(out_plot, f'diag_KO_permutation_norm{tag}.pdf'),
                formats=('pdf', 'png'))

    # ── size-normalised grouped heatmap (8 groups × 12 neuropils) ─────────────
    group_delta_norm = np.array([
        np.nanmean(delta_mean_norm[GROUPS[g], :], axis=0)
        for g in GROUP_ORDER
    ])
    df_group_norm = pd.DataFrame(group_delta_norm, index=GROUP_ORDER, columns=NEUROPILS)
    df_group_norm.to_csv(os.path.join(out_diag, 'KO_delta_accuracy_groups_norm.csv'))

    vmax_gn = np.nanpercentile(np.abs(group_delta_norm), 95)
    fig_gn, ax_gn = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        df_group_norm, ax=ax_gn, cmap='RdBu_r', center=0,
        vmin=-vmax_gn, vmax=vmax_gn,
        annot=True, fmt='.2f',
        annot_kws={'size': ANNOT_FS},
        linewidths=0.3,
        cbar_kws={'label': f'Mean ΔAccuracy per {SCALE:,} voxels [pp / 10k vox]', 'shrink': 0.8},
    )
    ax_gn.set_title(
        f'Group-level accuracy drop — size-normalised  (n = {len(all_deltas)} runs)',
        fontsize=TITLE_FS, pad=14,
    )
    ax_gn.set_xlabel('Neuropil knocked out', fontsize=LABEL_FS, labelpad=8)
    ax_gn.set_ylabel('Condition group',      fontsize=LABEL_FS, labelpad=8)
    ax_gn.tick_params(axis='both', labelsize=TICK_FS)
    cbar_gn = ax_gn.collections[0].colorbar
    cbar_gn.ax.tick_params(labelsize=CBAR_FS)
    cbar_gn.ax.yaxis.label.set_size(CBAR_FS)
    plt.tight_layout()
    save_figure(fig_gn, os.path.join(out_plot, f'diag_KO_permutation_groups_norm{tag}.pdf'),
                formats=('pdf', 'png'))

    # ── scatter: voxel size vs mean ΔAcc ──────────────────────────────────────
    mean_delta_per_neuropil = np.nanmean(delta_mean, axis=0)
    fig_sz, ax_sz = plt.subplots(figsize=(7, 6))
    ax_sz.scatter(neuropil_sizes / 1000, mean_delta_per_neuropil,
                  s=80, color='steelblue', edgecolors='white', linewidths=0.5, zorder=3)
    for name, x, y in zip(NEUROPILS, neuropil_sizes / 1000, mean_delta_per_neuropil):
        ax_sz.annotate(name, (x, y), textcoords='offset points', xytext=(6, 4),
                       fontsize=TICK_FS - 1)
    ax_sz.axhline(0, color='grey', linewidth=0.6, linestyle='--')
    ax_sz.set_xlabel('Neuropil 3D volume [× 1,000 voxels]', fontsize=LABEL_FS)
    ax_sz.set_ylabel('Mean ΔAccuracy (baseline − KO) [pp]', fontsize=LABEL_FS)
    ax_sz.set_title('Neuropil size vs accuracy drop\n(size confound diagnostic)',
                    fontsize=TITLE_FS, pad=10)
    ax_sz.tick_params(axis='both', labelsize=TICK_FS)
    plt.tight_layout()
    save_figure(fig_sz, os.path.join(out_plot, f'diag_KO_size_vs_delta{tag}.pdf'),
                formats=('pdf', 'png'))

    # ── 2D footprint size normalisation ───────────────────────────────────────
    neuropil_sizes_2d  = compute_neuropil_sizes_2d(base_config, rng=rng, logger=logger)
    SCALE_2D           = 1_000   # express as ΔAcc per 1k pixels
    delta_mean_norm_2d = delta_mean / (neuropil_sizes_2d / SCALE_2D)  # (n_classes, n_neuropils)

    df_norm_2d = pd.DataFrame(delta_mean_norm_2d, index=class_names, columns=NEUROPILS)
    df_norm_2d.to_csv(os.path.join(out_diag, 'KO_delta_accuracy_norm_2d.csv'))

    group_delta_norm_2d = np.array([
        np.nanmean(delta_mean_norm_2d[GROUPS[g], :], axis=0)
        for g in GROUP_ORDER
    ])
    df_group_norm_2d = pd.DataFrame(group_delta_norm_2d, index=GROUP_ORDER, columns=NEUROPILS)
    df_group_norm_2d.to_csv(os.path.join(out_diag, 'KO_delta_accuracy_groups_norm_2d.csv'))
    logger.info(f'Saved 2D-norm CSVs → {out_diag}')

    # heatmap: per-class 2D norm
    vmax_n2 = np.nanpercentile(np.abs(delta_mean_norm_2d), 95)
    fig_n2, ax_n2 = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        df_norm_2d, ax=ax_n2, cmap='RdBu_r', center=0,
        vmin=-vmax_n2, vmax=vmax_n2,
        annot=True, fmt='.2f',
        annot_kws={'size': ANNOT_FS},
        linewidths=0.3,
        cbar_kws={'label': f'Mean ΔAccuracy per {SCALE_2D:,} pixels [pp / 1k px]', 'shrink': 0.8},
    )
    ax_n2.set_title(
        f'Per-class accuracy drop — 2D footprint normalised  (n = {len(all_deltas)} runs)',
        fontsize=TITLE_FS, pad=14,
    )
    ax_n2.set_xlabel('Neuropil knocked out', fontsize=LABEL_FS, labelpad=8)
    ax_n2.set_ylabel('Behavioural class',    fontsize=LABEL_FS, labelpad=8)
    ax_n2.tick_params(axis='both', labelsize=TICK_FS)
    cbar_n2 = ax_n2.collections[0].colorbar
    cbar_n2.ax.tick_params(labelsize=CBAR_FS)
    cbar_n2.ax.yaxis.label.set_size(CBAR_FS)
    plt.tight_layout()
    save_figure(fig_n2, os.path.join(out_plot, f'diag_KO_permutation_norm_2d{tag}.pdf'),
                formats=('pdf', 'png'))

    # heatmap: group-level 2D norm
    vmax_gn2 = np.nanpercentile(np.abs(group_delta_norm_2d), 95)
    fig_gn2, ax_gn2 = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        df_group_norm_2d, ax=ax_gn2, cmap='RdBu_r', center=0,
        vmin=-vmax_gn2, vmax=vmax_gn2,
        annot=True, fmt='.2f',
        annot_kws={'size': ANNOT_FS},
        linewidths=0.3,
        cbar_kws={'label': f'Mean ΔAccuracy per {SCALE_2D:,} pixels [pp / 1k px]', 'shrink': 0.8},
    )
    ax_gn2.set_title(
        f'Group-level accuracy drop — 2D footprint normalised  (n = {len(all_deltas)} runs)',
        fontsize=TITLE_FS, pad=14,
    )
    ax_gn2.set_xlabel('Neuropil knocked out', fontsize=LABEL_FS, labelpad=8)
    ax_gn2.set_ylabel('Condition group',      fontsize=LABEL_FS, labelpad=8)
    ax_gn2.tick_params(axis='both', labelsize=TICK_FS)
    cbar_gn2 = ax_gn2.collections[0].colorbar
    cbar_gn2.ax.tick_params(labelsize=CBAR_FS)
    cbar_gn2.ax.yaxis.label.set_size(CBAR_FS)
    plt.tight_layout()
    save_figure(fig_gn2, os.path.join(out_plot, f'diag_KO_permutation_groups_norm_2d{tag}.pdf'),
                formats=('pdf', 'png'))

    # scatter: 2D pixel footprint vs mean ΔAcc
    mean_delta_per_neuropil = np.nanmean(delta_mean, axis=0)
    fig_sz2, ax_sz2 = plt.subplots(figsize=(7, 6))
    ax_sz2.scatter(neuropil_sizes_2d, mean_delta_per_neuropil,
                   s=80, color='steelblue', edgecolors='white', linewidths=0.5, zorder=3)
    for name, x, y in zip(NEUROPILS, neuropil_sizes_2d, mean_delta_per_neuropil):
        ax_sz2.annotate(name, (x, y), textcoords='offset points', xytext=(6, 4),
                        fontsize=TICK_FS - 1)
    ax_sz2.axhline(0, color='grey', linewidth=0.6, linestyle='--')
    ax_sz2.set_xlabel('Neuropil 2D footprint [pixels]', fontsize=LABEL_FS)
    ax_sz2.set_ylabel('Mean ΔAccuracy (baseline − KO) [pp]', fontsize=LABEL_FS)
    ax_sz2.set_title('Neuropil 2D size vs accuracy drop\n(size confound diagnostic)',
                     fontsize=TITLE_FS, pad=10)
    ax_sz2.tick_params(axis='both', labelsize=TICK_FS)
    plt.tight_layout()
    save_figure(fig_sz2, os.path.join(out_plot, f'diag_KO_size2d_vs_delta{tag}.pdf'),
                formats=('pdf', 'png'))

    # contrast plots: State / Modality / Valence from 2D-norm group deltas
    _CONTRAST_PAIRS = {
        'Starved_minus_Fed':         ('Starved',    'Fed'),
        'Odor_minus_Taste':          ('Odor',       'Taste'),
        'Appetitive_minus_Aversive': ('Appetitive', 'Aversive'),
    }
    contrast_rows_2d = {
        cname: df_group_norm_2d.loc[g1] - df_group_norm_2d.loc[g2]
        for cname, (g1, g2) in _CONTRAST_PAIRS.items()
        if g1 in df_group_norm_2d.index and g2 in df_group_norm_2d.index
    }
    contrasts_df_2d = pd.DataFrame(contrast_rows_2d).T   # (3, n_neuropils)
    contrasts_df_2d.to_csv(os.path.join(out_diag, 'KO_contrasts_norm_2d.csv'))

    fig_c, axes_c = plt.subplots(1, 3, figsize=(11, 3.5))
    plot_contrasts_horizontal(
        axes=axes_c,
        group_contrasts_abs=contrasts_df_2d,
        neuropil_names=NEUROPILS,
        sort_by_modality=True,
        uniform_xlim=True,
        fontsize_title=FONT_SIZES.get('subplot_title', 9),
        fontsize_labels=FONT_SIZES.get('label', 7),
    )
    fig_c.suptitle(
        f'KO neuropil importance — 2D footprint normalised contrasts  (n = {len(all_deltas)} runs)',
        fontsize=TITLE_FS - 4, y=1.02,
    )
    plt.tight_layout()
    save_figure(fig_c, os.path.join(out_plot, f'diag_KO_contrasts_norm_2d{tag}.pdf'),
                formats=('pdf', 'png'))

    logger.info('Done.')


if __name__ == '__main__':
    main()
