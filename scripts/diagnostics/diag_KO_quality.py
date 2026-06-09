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
    python scripts/diagnostics/diag_KO_quality.py

Outputs:
    stdout                                    — structured per-neuropil report
    results/diagnostics/KO_quality/summary.csv
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config_loader import load_config

# ── constants ─────────────────────────────────────────────────────────────────

NEUROPILS = ['AL', 'MB', 'PENP', 'VLNP', 'CX', 'GNG',
             'LX', 'SNP', 'INP', 'LH', 'OL', 'VMNP']
N_SAMPLE    = 5   # recordings sampled per neuropil for pixel stats
N_FRAMES    = 3   # frames per recording
RANDOM_SEED = 42

OUT_DIR = 'results/diagnostics/KO_quality'


# ── helpers ───────────────────────────────────────────────────────────────────

def ko_path(baseline_path: Path, neuropil: str) -> Path:
    return baseline_path.parent / f'meanZ_allTs_KO_{neuropil}'


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


def section(title: str):
    print(f'\n{"─" * 64}')
    print(f'  {title}')
    print(f'{"─" * 64}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    config       = load_config()
    baseline_dir = Path(config['paths']['allTs_path'])

    baseline_recs = {
        p.name for p in baseline_dir.iterdir()
        if p.is_dir() and any(p.glob('*.tiff'))
    }
    print(f'Baseline : {baseline_dir}')
    print(f'Recordings in baseline: {len(baseline_recs)}')

    rows = []

    for neuropil in NEUROPILS:
        ko_dir = ko_path(baseline_dir, neuropil)
        section(f'{neuropil}  →  {ko_dir}')

        # ── 1. Completeness ───────────────────────────────────────────────────
        if not ko_dir.exists():
            print('  [MISSING] KO directory does not exist — needs full preprocessing')
            rows.append({'neuropil': neuropil, 'status': 'MISSING_DIR'})
            continue

        ko_recs  = {
            p.name for p in ko_dir.iterdir()
            if p.is_dir() and any(p.glob('*.tiff'))
        }
        missing  = sorted(baseline_recs - ko_recs)
        extra    = sorted(ko_recs - baseline_recs)

        print(f'  KO recordings : {len(ko_recs)} / {len(baseline_recs)} baseline')
        if missing:
            preview = ', '.join(missing[:8]) + (f'  … (+{len(missing)-8} more)' if len(missing) > 8 else '')
            print(f'  [WARNING] {len(missing)} recordings missing from KO: {preview}')
        else:
            print('  Completeness  : OK')
        if extra:
            print(f'  [INFO] {len(extra)} recordings in KO not in baseline: {", ".join(extra[:5])}')

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
            print(f'  [WARNING] Frame count mismatches in sampled {N_SAMPLE} recordings:')
            for rec, nb, nk in mismatches:
                print(f'    {rec}: baseline={nb}  KO={nk}')
        else:
            print(f'  Frame counts  : OK (checked {len(sample)} recordings)')

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
            print('  [ERROR] Could not load any frames.')
            rows.append({'neuropil': neuropil, 'status': 'NO_FRAMES'})
            continue

        n_rec = len(base_stats_all)
        print(f'\n  Pixel stats (avg over {n_rec} recordings × {N_FRAMES} frames):')
        print(f'  {"Metric":<20} {"Baseline":>12} {"KO":>12} {"Δ (KO−B)":>12}')
        print(f'  {"─"*20} {"─"*12} {"─"*12} {"─"*12}')
        for key in ['min', 'max', 'mean', 'std', 'zero_frac']:
            bv = avg(base_stats_all, key)
            kv = avg(ko_stats_all,   key)
            print(f'  {key:<20} {bv:>12.4f} {kv:>12.4f} {kv - bv:>+12.4f}')

        mean_delta = float(np.mean(delta_signal_all)) if delta_signal_all else float('nan')
        print(f'  {"Δ signal (B−KO)":<20} {mean_delta:>12.4f}')

        if nan_inf_total > 0:
            print(f'  [WARNING] {nan_inf_total} NaN/Inf pixels detected across sample')
        else:
            print('  NaN/Inf       : none detected')

        rows.append({
            'neuropil':           neuropil,
            'status':             'OK' if not missing else 'INCOMPLETE',
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
    print(f'\n{"═" * 64}')
    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, 'summary.csv')
    df.to_csv(out_csv, index=False)
    print(f'Summary CSV saved → {out_csv}')

    # quick status table
    print(f'\n{"Neuropil":<8} {"Status":<12} {"Missing":>8} {"ΔZero%":>10} {"ΔSignal":>10}')
    print('─' * 52)
    for _, r in df.iterrows():
        if r.get('status') in ('MISSING_DIR', 'NO_FRAMES'):
            print(f'{r["neuropil"]:<8} {r["status"]:<12}')
        else:
            dz = r.get('delta_zero_frac', float('nan'))
            ds = r.get('delta_signal', float('nan'))
            print(f'{r["neuropil"]:<8} {r["status"]:<12} {int(r["n_missing"]):>8} {dz*100:>+9.2f}% {ds:>10.4f}')


if __name__ == '__main__':
    main()
