"""
Diagnostic: KO preprocessing quality check
==========================================
For each neuropil KO directory:
  1. Completeness  — which recordings are missing vs baseline
  2. Frame count   — do present recordings match baseline frame count?
  3. Pixel stats   — min, max, mean, std (baseline vs KO, sampled)
  4. Zero fraction — fraction of zero pixels (baseline vs KO)
  5. Δ signal      — mean(baseline − KO) per recording
  6. NaN / Inf     — data sanity check

Usage (from repo root):
    python -m scripts.analysis.diag_KO_quality

Outputs:
    stdout                                    — structured per-neuropil report
    results/diagnostics/KO_quality/summary.csv
"""

import argparse
import os
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

# ── constants ─────────────────────────────────────────────────────────────────

NEUROPILS = ['AL', 'MB', 'PENP', 'VLNP', 'CX', 'GNG',
             'LX', 'SNP', 'INP', 'LH', 'OL', 'VMNP']
N_SAMPLE    = 5   # recordings sampled per neuropil for pixel stats
N_FRAMES    = 3   # frames per recording
N_VIZ       = 10  # example frames for KO image grid
RANDOM_SEED = 42

OUT_DIR = 'results/diagnostics/KO_quality'


# ── helpers ───────────────────────────────────────────────────────────────────

def ko_path(baseline_path: Path, neuropil: str, variant: str = '') -> Path:
    suffix = f'_{variant}' if variant in ('noisefill', 'static', 'shuffled') else ''
    return baseline_path.parent / f'meanZ_allTs_KO{suffix}_{neuropil}'


def load_frames(rec_dir: Path, n: int, rng: random.Random):
    tiffs = sorted(rec_dir.glob('*.tiff'))
    if not tiffs:
        return None
    chosen = rng.sample(tiffs, min(n, len(tiffs)))
    try:
        return np.stack([tifffile.imread(str(t)).astype(np.float32) for t in chosen])
    except Exception:
        return None


def pixel_stats(arr: np.ndarray) -> dict:
    flat = arr.ravel()
    valid = flat[np.isfinite(flat)]
    return {
        'min':       float(valid.min())                      if valid.size else float('nan'),
        'max':       float(valid.max())                      if valid.size else float('nan'),
        'mean':      float(valid.mean())                     if valid.size else float('nan'),
        'std':       float(valid.std())                      if valid.size else float('nan'),
        'zero_frac': float((flat == 0).sum() / flat.size),
        'nan_inf':   int((~np.isfinite(flat)).sum()),
    }


def avg(stat_list: list, key: str) -> float:
    vals = [d[key] for d in stat_list if np.isfinite(d[key])]
    return float(np.mean(vals)) if vals else float('nan')


def section(title: str, logger):
    logger.info('─' * 64)
    logger.info(f'  {title}')
    logger.info('─' * 64)


def plot_ko_image_grid(baseline_dir: Path, neuropils: list, n_examples: int,
                       rng: random.Random, out_dir: str, variant: str = ''):
    """Grid of n_examples columns × 13 rows (original + 12 KOs)."""
    # collect all (recording, tiff_path) pairs
    all_frames = [
        tiff
        for rec_dir in sorted(baseline_dir.iterdir())
        if rec_dir.is_dir() and any(rec_dir.glob('*.tiff'))
        for tiff in sorted(rec_dir.glob('*.tiff'))
    ]
    if not all_frames:
        print('  [WARNING] No baseline frames found for visualization.')
        return

    sample = rng.sample(all_frames, min(n_examples, len(all_frames)))
    row_labels = ['Original'] + [f'KO {n}' for n in neuropils]
    n_rows = len(row_labels)
    n_cols = len(sample)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.8, n_rows * 1.8),
        gridspec_kw={'hspace': 0.04, 'wspace': 0.04},
    )
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col, frame_path in enumerate(sample):
        rec_name = frame_path.parent.name
        frame_name = frame_path.name

        img_base = tifffile.imread(str(frame_path)).astype(np.float32)
        vmax = img_base.max() if img_base.max() > 0 else 1.0

        for row, label in enumerate(row_labels):
            ax = axes[row, col]

            if row == 0:
                img = img_base
                ax.set_title(f'{rec_name}\nt={frame_path.stem.split("_")[-1]}',
                             fontsize=5, pad=2)
            else:
                neuropil = neuropils[row - 1]
                ko_suffix = '_noisefill' if variant == 'noisefill' else ''
                ko_path  = baseline_dir.parent / f'meanZ_allTs_KO{ko_suffix}_{neuropil}' / rec_name / frame_name
                if ko_path.exists():
                    img = tifffile.imread(str(ko_path)).astype(np.float32)
                else:
                    img = None

            if img is not None:
                ax.imshow(img, cmap='gray', vmin=0, vmax=vmax, aspect='auto')
            else:
                ax.set_facecolor('#444444')
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=5, color='white')

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if col == 0:
                ax.text(-0.12, 0.5, label, transform=ax.transAxes,
                        fontsize=6, ha='right', va='center', rotation=0)

    fig.suptitle(f'Example frames: original vs neuropil KOs  (n={n_cols})',
                 fontsize=9, y=1.002)

    out_path = os.path.join(out_dir, 'ko_image_examples.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    # caller's logger not in scope here — use print for this one line
    print(f'KO image grid saved → {out_path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='',
                        help='KO variant to check: empty = zero-fill (default), noisefill, static')
    args = parser.parse_args()
    variant = args.variant

    out_dir = OUT_DIR + (f'_{variant}' if variant else '')
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    log_tag = f'diag_KO_quality{"_" + variant if variant else ""}'
    logger  = setup_logger(task_name=log_tag, log_dir='logs/diag_KO_quality')

    config       = load_config()
    baseline_dir = Path(config['paths']['allTs_path'])

    baseline_recs = {
        p.name for p in baseline_dir.iterdir()
        if p.is_dir() and any(p.glob('*.tiff'))
    }
    logger.info(f'Variant  : {variant if variant else "zero-fill (default)"}')
    logger.info(f'Baseline : {baseline_dir}')
    logger.info(f'Recordings in baseline: {len(baseline_recs)}')

    rows = []

    for neuropil in NEUROPILS:
        ko_dir = ko_path(baseline_dir, neuropil, variant)
        section(f'{neuropil}  →  {ko_dir}', logger)

        # ── 1. Completeness ───────────────────────────────────────────────────
        if not ko_dir.exists():
            logger.warning('  [MISSING] KO directory does not exist — needs full preprocessing')
            rows.append({'neuropil': neuropil, 'status': 'MISSING_DIR'})
            continue

        ko_recs  = {
            p.name for p in ko_dir.iterdir()
            if p.is_dir() and any(p.glob('*.tiff'))
        }
        missing  = sorted(baseline_recs - ko_recs)
        extra    = sorted(ko_recs - baseline_recs)

        logger.info(f'  KO recordings : {len(ko_recs)} / {len(baseline_recs)} baseline')
        if missing:
            preview = ', '.join(missing[:8]) + (f'  … (+{len(missing)-8} more)' if len(missing) > 8 else '')
            logger.warning(f'  {len(missing)} recordings missing from KO: {preview}')
        else:
            logger.info('  Completeness  : OK')
        if extra:
            logger.info(f'  {len(extra)} recordings in KO not in baseline: {", ".join(extra[:5])}')

        # ── 2. Frame count ────────────────────────────────────────────────────
        common = sorted(baseline_recs & ko_recs)
        sample = rng.sample(common, min(N_SAMPLE, len(common)))

        mismatches = []
        for rec in sample:
            nb = len(list((baseline_dir / rec).glob('*.tiff')))
            nk = len(list((ko_dir / rec).glob('*.tiff')))
            if nb != nk:
                mismatches.append((rec, nb, nk))

        if mismatches:
            logger.warning(f'  Frame count mismatches in sampled {N_SAMPLE} recordings:')
            for rec, nb, nk in mismatches:
                logger.warning(f'    {rec}: baseline={nb}  KO={nk}')
        else:
            logger.info(f'  Frame counts  : OK (checked {len(sample)} recordings)')

        # ── 3–6. Pixel stats, zero fraction, Δ signal, NaN/Inf ───────────────
        base_stats_all, ko_stats_all, delta_signal_all = [], [], []
        nan_inf_total = 0

        for rec in sample:
            bf = load_frames(baseline_dir / rec, N_FRAMES, rng)
            kf = load_frames(ko_dir / rec,       N_FRAMES, rng)
            if bf is None or kf is None:
                continue

            bs = pixel_stats(bf)
            ks = pixel_stats(kf)
            base_stats_all.append(bs)
            ko_stats_all.append(ks)
            nan_inf_total += bs['nan_inf'] + ks['nan_inf']

            n = min(bf.shape[0], kf.shape[0])
            delta_signal_all.append(float(np.mean(bf[:n] - kf[:n])))

        if not base_stats_all:
            logger.error('  Could not load any frames.')
            rows.append({'neuropil': neuropil, 'status': 'NO_FRAMES'})
            continue

        # structural distance (image-level OOD proxy)
        l2_dists, abs_diffs = [], []
        for bs_arr, ks_arr in zip(
            [load_frames(baseline_dir / rec, N_FRAMES, rng) for rec in sample],
            [load_frames(ko_dir / rec,       N_FRAMES, rng) for rec in sample],
        ):
            if bs_arr is None or ks_arr is None:
                continue
            n = min(bs_arr.shape[0], ks_arr.shape[0])
            diff = bs_arr[:n] - ks_arr[:n]
            l2_dists.append(float(np.linalg.norm(diff) / diff.size))
            abs_diffs.append(float(np.mean(np.abs(diff))))
        mean_l2   = float(np.mean(l2_dists))  if l2_dists  else float('nan')
        mean_abs  = float(np.mean(abs_diffs)) if abs_diffs else float('nan')

        n_rec = len(base_stats_all)
        logger.info(f'  Pixel stats (avg over {n_rec} recordings × {N_FRAMES} frames):')
        logger.info(f'  {"Metric":<20} {"Baseline":>12} {"KO":>12} {"Δ (KO−B)":>12}')
        for key in ['min', 'max', 'mean', 'std', 'zero_frac']:
            bv = avg(base_stats_all, key)
            kv = avg(ko_stats_all,   key)
            logger.info(f'  {key:<20} {bv:>12.4f} {kv:>12.4f} {kv - bv:>+12.4f}')

        mean_delta = float(np.mean(delta_signal_all)) if delta_signal_all else float('nan')
        logger.info(f'  {"Δ signal (B−KO)":<20} {mean_delta:>12.4f}')
        logger.info(f'  {"L2 dist (norm)":<20} {mean_l2:>12.6f}')
        logger.info(f'  {"Mean |Δ| pixel":<20} {mean_abs:>12.6f}')

        if nan_inf_total > 0:
            logger.warning(f'  {nan_inf_total} NaN/Inf pixels detected across sample')
        else:
            logger.info('  NaN/Inf       : none detected')

        rows.append({
            'neuropil':           neuropil,
            'status':             'OK' if not missing else 'INCOMPLETE',
            'mean_l2_dist':       mean_l2,
            'mean_abs_diff':      mean_abs,
            'n_baseline_recs':    len(baseline_recs),
            'n_ko_recs':          len(ko_recs),
            'n_missing':          len(missing),
            'n_frame_mismatches': len(mismatches),
            'baseline_min':       avg(base_stats_all, 'min'),
            'baseline_max':       avg(base_stats_all, 'max'),
            'baseline_mean':      avg(base_stats_all, 'mean'),
            'baseline_std':       avg(base_stats_all, 'std'),
            'baseline_zero_frac': avg(base_stats_all, 'zero_frac'),
            'ko_min':             avg(ko_stats_all, 'min'),
            'ko_max':             avg(ko_stats_all, 'max'),
            'ko_mean':            avg(ko_stats_all, 'mean'),
            'ko_std':             avg(ko_stats_all, 'std'),
            'ko_zero_frac':       avg(ko_stats_all, 'zero_frac'),
            'delta_zero_frac':    avg(ko_stats_all, 'zero_frac') - avg(base_stats_all, 'zero_frac'),
            'delta_signal':       mean_delta,
            'nan_inf_count':      nan_inf_total,
        })

    # ── summary ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, 'summary.csv')
    df.to_csv(out_csv, index=False)
    logger.info(f'Summary CSV saved → {out_csv}')

    logger.info(f'{"Neuropil":<8} {"Status":<12} {"Missing":>8} {"ΔZero%":>10} {"ΔSignal":>10}')
    for _, r in df.iterrows():
        if r.get('status') in ('MISSING_DIR', 'NO_FRAMES'):
            logger.warning(f'{r["neuropil"]:<8} {r["status"]:<12}')
        else:
            dz = r.get('delta_zero_frac', float('nan'))
            ds = r.get('delta_signal', float('nan'))
            logger.info(f'{r["neuropil"]:<8} {r["status"]:<12} {int(r["n_missing"]):>8} {dz*100:>+9.2f}% {ds:>10.4f}')

    # ── structural distance plot ───────────────────────────────────────────────
    df_complete = df.dropna(subset=['mean_l2_dist'])
    if not df_complete.empty:
        fig_sd, axes_sd = plt.subplots(1, 2, figsize=(13, 4))

        for ax, col, ylabel, title in [
            (axes_sd[0], 'mean_l2_dist',  'Normalised L2 distance',  'Structural distance (L2)'),
            (axes_sd[1], 'mean_abs_diff', 'Mean |baseline − KO|', 'Mean absolute pixel change'),
        ]:
            vals = df_complete[col].values
            neuropils_plot = df_complete['neuropil'].values
            colors = plt.cm.RdYlGn_r(vals / vals.max())
            ax.bar(neuropils_plot, vals, color=colors, edgecolor='white', linewidth=0.5)
            ax.set_xlabel('Neuropil knocked out', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=11, pad=8)
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            ax.tick_params(axis='y', labelsize=9)

        plt.suptitle('Image-level structural change per neuropil KO\n'
                     '(larger = more OOD-like input to model)',
                     fontsize=12, y=1.02)
        plt.tight_layout()
        out_sd = os.path.join(out_dir, 'ko_structural_distance.png')
        fig_sd.savefig(out_sd, dpi=150, bbox_inches='tight')
        plt.close(fig_sd)
        logger.info(f'Structural distance plot saved → {out_sd}')

    # ── KO image grid ─────────────────────────────────────────────────────────
    logger.info(f'Generating KO image grid ({N_VIZ} examples × {len(NEUROPILS) + 1} rows)...')
    plot_ko_image_grid(baseline_dir, NEUROPILS, N_VIZ, rng, out_dir, variant)


if __name__ == '__main__':
    main()
