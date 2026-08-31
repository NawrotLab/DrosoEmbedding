"""
Diagnostic: KO data coverage check
====================================
For each neuropil KO directory, checks which test set files are missing.
No model loading — pure file existence check.

Usage (from repo root):
    python -m scripts.analysis.diag_KO_data_coverage

Outputs:
    results/diagnostics/KO_permutation/KO_data_coverage.csv
    — rows = neuropils, columns: total_files, present, missing, pct_missing
    results/diagnostics/KO_permutation/KO_missing_files.txt
    — full list of missing files per neuropil
"""

import os
import sys
import pickle
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config_loader import load_config
from src.utils.helpers import paths2neuropilpaths

# ── constants ─────────────────────────────────────────────────────────────────

NEUROPILS = ['AL', 'MB', 'PENP', 'VLNP', 'CX', 'GNG',
             'LX', 'SNP', 'INP', 'LH', 'OL', 'VMNP']

OUT_DIR = 'results/diagnostics/KO_permutation'


def ko_allTs_path(base_allTs_path: str, neuropil: str) -> str:
    parent = os.path.dirname(base_allTs_path.rstrip('/'))
    return os.path.join(parent, f'meanZ_allTs_KO_{neuropil}')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    config = load_config()
    with open(f"{config['paths']['results_root']}/config.pkl", 'rb') as fh:
        train_config = pickle.load(fh)

    allTs_base = train_config['paths']['allTs_path']

    with open(config['paths']['pickle_path'], 'rb') as fh:
        _, _, X_test, _, _, Y_test = pickle.load(fh)

    print(f"Test set: {len(X_test)} files\n")

    summary_rows = []
    missing_log  = []

    for neuropil in NEUROPILS:
        allTs_ko = ko_allTs_path(allTs_base, neuropil)

        if not Path(allTs_ko).exists():
            print(f'[{neuropil}]  directory missing: {allTs_ko}')
            summary_rows.append({
                'neuropil':    neuropil,
                'directory':   allTs_ko,
                'dir_exists':  False,
                'total_files': len(X_test),
                'present':     0,
                'missing':     len(X_test),
                'pct_missing': 100.0,
            })
            missing_log.append(f'\n=== {neuropil} — DIRECTORY MISSING: {allTs_ko} ===')
            continue

        # build KO paths by replacing base allTs dir in each path
        cfg_ko = {'data': {'preprocessing': {
            'remove_neuropil': True,
            'isolate_neuropil': False,
            'neuropil': neuropil,
        }}, 'paths': {'allTs_path': allTs_ko}}
        X_ko = paths2neuropilpaths(list(X_test), cfg_ko)

        missing = [p for p in X_ko if not Path(p).exists()]
        present = len(X_ko) - len(missing)
        pct     = 100.0 * len(missing) / len(X_ko)

        print(f'[{neuropil}]  present={present}/{len(X_ko)}  missing={len(missing)} ({pct:.1f}%)')

        summary_rows.append({
            'neuropil':    neuropil,
            'directory':   allTs_ko,
            'dir_exists':  True,
            'total_files': len(X_ko),
            'present':     present,
            'missing':     len(missing),
            'pct_missing': round(pct, 2),
        })

        if missing:
            missing_log.append(f'\n=== {neuropil} ({len(missing)} missing) ===')
            missing_log.extend(missing)

    # ── save summary CSV ──────────────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(OUT_DIR, 'KO_data_coverage.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nSaved summary → {csv_path}')

    # ── save missing files list ───────────────────────────────────────────────
    txt_path = os.path.join(OUT_DIR, 'KO_missing_files.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(missing_log))
    print(f'Saved missing files → {txt_path}')


if __name__ == '__main__':
    main()
