"""
Diagnostic: KO preprocessing pixel statistics
==============================================
Investigates why GNG produces massive accuracy drops in the KO heatmap.

For each neuropil KO, samples N_SAMPLE test recordings and reports:
  1. Directory existence + total TIFF count
  2. Fraction of exactly-zero pixels (KO vs baseline, and delta)
  3. Mean / std of non-zero pixel signal
  4. Largest contiguous zero region + number of zero connected components
     (contiguity proxy: a single large block = OOD; many small holes = OK)

Usage (from repo root):
    python scripts/diagnostics/diag_preprocessing_KO.py

Output: structured text → stdout (paste directly into conversation)
"""

import os
import sys
import pickle
import random
import numpy as np
import tifffile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config_loader import load_config

# ── constants ─────────────────────────────────────────────────────────────────
NEUROPILS = ['AL', 'MB', 'PENP', 'VLNP', 'CX', 'GNG',
             'LX', 'SNP', 'INP', 'LH', 'OL', 'VMNP']

N_SAMPLE = 5   # recordings to sample per neuropil
N_FRAMES = 3   # frames per recording (averaged for stats)

random.seed(42)


# ── helpers ───────────────────────────────────────────────────────────────────

def ko_allTs_path(base: str, neuropil: str) -> str:
    parent = os.path.dirname(base.rstrip('/'))
    return os.path.join(parent, f'meanZ_allTs_KO_{neuropil}')


def pixel_stats(img: np.ndarray) -> dict:
    flat = img.ravel().astype(np.float64)
    n_total = flat.size
    n_zero  = int((flat == 0.0).sum())
    nonzero = flat[flat > 0.0]
    return {
        'frac_zero': n_zero / n_total,
        'mean_nz':   float(nonzero.mean()) if nonzero.size else float('nan'),
        'std_nz':    float(nonzero.std())  if nonzero.size else float('nan'),
        'max':       float(flat.max()),
        'shape':     img.shape,
    }


def largest_zero_component(img: np.ndarray):
    """Return (size_of_largest_zero_blob, n_blobs). Needs scipy."""
    try:
        from scipy.ndimage import label
        labeled, n = label(img == 0.0)
        if n == 0:
            return 0, 0
        sizes = np.bincount(labeled.ravel())[1:]
        return int(sizes.max()), int(n)
    except ImportError:
        return -1, -1


def load_frames(allTs_dir: str, recording: str, start_frame: int, n_frames: int):
    """Return list of 2D float32 arrays for frames [start, start+n_frames)."""
    imgs = []
    for f in range(start_frame, start_frame + n_frames):
        p = Path(allTs_dir) / recording / f'{recording}_{f}.tiff'
        if p.exists():
            imgs.append(tifffile.imread(str(p)).astype(np.float32))
    return imgs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    config = load_config()
    with open(f"{config['paths']['results_root']}/config.pkl", 'rb') as fh:
        train_config = pickle.load(fh)
    allTs_base = train_config['paths']['allTs_path']

    with open(config['paths']['pickle_path'], 'rb') as fh:
        _, _, X_test, _, _, _ = pickle.load(fh)

    print('=' * 72)
    print('DIAGNOSTIC: KO preprocessing pixel statistics')
    print('=' * 72)
    print(f'\nBaseline allTs : {allTs_base}')
    print(f'Test set paths : {len(X_test)}')

    # Extract unique (recording, start_frame) pairs
    entries = set()
    for p in X_test:
        parts = Path(p).parts
        rec = parts[-2]
        try:
            sf = int(Path(p).stem.split('_')[-1])
        except ValueError:
            continue
        entries.add((rec, sf))
    entries = list(entries)
    print(f'Unique (rec, frame) pairs: {len(entries)}\n')

    # ── baseline stats ────────────────────────────────────────────────────────
    print('─' * 72)
    print('BASELINE (no KO) — sampled pixel statistics')
    print('─' * 72)
    sample_base = random.sample(entries, min(N_SAMPLE, len(entries)))
    base_frac_zeros = []
    base_mean_nz    = []
    example_shape   = None

    for rec, sf in sample_base:
        imgs = load_frames(allTs_base, rec, sf, N_FRAMES)
        for img in imgs:
            s = pixel_stats(img)
            if example_shape is None:
                example_shape = s['shape']
            base_frac_zeros.append(s['frac_zero'])
            base_mean_nz.append(s['mean_nz'])
            print(f'  {rec}/{rec}_{sf}  shape={s["shape"]}  '
                  f'zero%={100*s["frac_zero"]:.2f}  '
                  f'mean_nz={s["mean_nz"]:.5f}  max={s["max"]:.4f}')

    bmfz = np.nanmean(base_frac_zeros)
    print(f'\n  → Baseline mean zero%: {100*bmfz:.2f} ± {100*np.nanstd(base_frac_zeros):.2f}')
    print(f'  → Baseline mean non-zero signal: {np.nanmean(base_mean_nz):.5f}\n')
    if example_shape:
        print(f'  → Image pixel count: {example_shape[0]}×{example_shape[1]} = {example_shape[0]*example_shape[1]:,}')

    # ── per-neuropil KO stats ─────────────────────────────────────────────────
    summary_rows = []

    for neuropil in NEUROPILS:
        allTs_ko = ko_allTs_path(allTs_base, neuropil)
        print('\n' + '─' * 72)
        print(f'NEUROPIL: {neuropil}')
        print(f'KO dir  : {allTs_ko}')

        if not Path(allTs_ko).exists():
            print('  !! DIRECTORY MISSING — skipped')
            summary_rows.append({'neuropil': neuropil, 'status': 'MISSING'})
            continue

        n_files = sum(1 for _ in Path(allTs_ko).rglob('*.tiff'))
        ko_recs = {p.name for p in Path(allTs_ko).iterdir() if p.is_dir()}
        valid   = [(r, sf) for r, sf in entries if r in ko_recs]
        print(f'  TIFF files in KO dir: {n_files}  |  recordings: {len(ko_recs)}')
        print(f'  Test entries matched: {len(valid)} / {len(entries)}')

        if not valid:
            print('  !! No matching test recordings — skipped')
            summary_rows.append({'neuropil': neuropil, 'status': 'NO_MATCH'})
            continue

        sample = random.sample(valid, min(N_SAMPLE, len(valid)))
        ko_frac_zeros, ko_mean_nz, zero_block_sizes, n_blobs_list = [], [], [], []

        print(f'  Sampling {len(sample)} recordings × up to {N_FRAMES} frames:')
        for rec, sf in sample:
            imgs_ko   = load_frames(allTs_ko,   rec, sf, N_FRAMES)
            imgs_base = load_frames(allTs_base, rec, sf, N_FRAMES)
            for img_ko, img_base in zip(imgs_ko, imgs_base):
                sk = pixel_stats(img_ko)
                sb = pixel_stats(img_base)
                delta = sk['frac_zero'] - sb['frac_zero']
                ko_frac_zeros.append(sk['frac_zero'])
                ko_mean_nz.append(sk['mean_nz'])
                biggest, n_blobs = largest_zero_component(img_ko)
                zero_block_sizes.append(biggest)
                n_blobs_list.append(n_blobs)
                print(f'    {rec}_{sf}:  '
                      f'KO zero%={100*sk["frac_zero"]:.1f}  '
                      f'base zero%={100*sb["frac_zero"]:.1f}  '
                      f'Δ={100*delta:+.1f}pp  '
                      f'mean_nz={sk["mean_nz"]:.5f}  '
                      f'largest_0_blob={biggest}px  n_blobs={n_blobs}')

        mean_ko_zero  = np.nanmean(ko_frac_zeros)
        delta_vs_base = mean_ko_zero - bmfz
        mean_nz       = np.nanmean(ko_mean_nz)
        mean_block    = np.nanmean(zero_block_sizes)
        max_block     = np.nanmax(zero_block_sizes) if zero_block_sizes else 0
        total_px      = example_shape[0] * example_shape[1] if example_shape else 1

        print(f'\n  SUMMARY {neuropil}:')
        print(f'    KO mean zero%        : {100*mean_ko_zero:.2f} ± {100*np.nanstd(ko_frac_zeros):.2f}')
        print(f'    Δ from baseline      : {100*delta_vs_base:+.2f} pp')
        print(f'    Fraction of image KO : {100*(mean_ko_zero - bmfz):.1f}% of pixels newly zeroed')
        print(f'    KO mean non-zero sig : {mean_nz:.5f}  (baseline: {np.nanmean(base_mean_nz):.5f})')
        print(f'    Largest zero blob    : {mean_block:.0f} px mean  /  {max_block} px max  '
              f'(image is {total_px:,} px total → {100*max_block/total_px:.1f}%)')
        print(f'    Num zero blobs       : {np.nanmean(n_blobs_list):.0f} mean')

        summary_rows.append({
            'neuropil':          neuropil,
            'status':            'OK',
            'ko_zero_pct_mean':  round(100 * mean_ko_zero, 2),
            'delta_vs_base_pp':  round(100 * delta_vs_base, 2),
            'ko_mean_nz_signal': round(mean_nz, 5),
            'largest_blob_px':   int(max_block),
            'largest_blob_pct':  round(100 * max_block / total_px, 1) if total_px else 0,
        })

    # ── summary table ─────────────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('SUMMARY TABLE')
    print('=' * 72)
    header = f"{'Neuropil':8}  {'KO zero%':>10}  {'Δ(pp)':>8}  {'mean_nz':>10}  {'biggest_blob%':>14}  {'status':>10}"
    print(header)
    print('-' * len(header))
    for r in summary_rows:
        if r['status'] != 'OK':
            print(f"{r['neuropil']:8}  {'—':>10}  {'—':>8}  {'—':>10}  {'—':>14}  {r['status']:>10}")
        else:
            print(f"{r['neuropil']:8}  {r['ko_zero_pct_mean']:>10.2f}  "
                  f"{r['delta_vs_base_pp']:>+8.2f}  "
                  f"{r['ko_mean_nz_signal']:>10.5f}  "
                  f"{r['largest_blob_pct']:>14.1f}  "
                  f"{'OK':>10}")

    print('\nBaseline zero%: {:.2f}  |  Baseline mean_nz: {:.5f}'.format(
        100 * bmfz, np.nanmean(base_mean_nz)))
    print('\nINTERPRETATION GUIDE:')
    print('  Δ(pp)          — extra zero% added by KO; larger = bigger neuropil mask')
    print('  biggest_blob%  — largest contiguous zero region as % of image;')
    print('                   high value → large OOD patch the model never saw during training')
    print('  mean_nz signal — if much lower than baseline, signal is concentrated in KO region')
    print('\nDONE')


if __name__ == '__main__':
    main()
