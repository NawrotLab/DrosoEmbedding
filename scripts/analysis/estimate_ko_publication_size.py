"""
Read-only sizing report for publishing neuropil-KO frame data.

sfig_ko_neuropils.py (the only figure that uses KO data) only ever reads:
  - the State_Modality_Valence_16 task's TEST set (not train/val, not the
    other two tasks)
  - the 'static' KO variant only (not noisefill/shuffled)
  - for each test anchor, a 5-frame sequence (seq_len=5, seq_steps=10:
    frames start, start+10, start+20, start+30, start+40) -- not just the
    anchor frame, and not the rest of that recording

This reports, per neuropil:
  - whether meanZ_allTs_KO_static_{neuropil} exists and its full on-disk
    size (du -sh) -- the "ship everything" number
  - how many of the test set's required (recording, frame) sequences are
    actually present, and an estimated size if we packaged ONLY those
    files -- the "ship exactly what the figure reads" number

Read-only: does not write, move, or delete anything.

Usage (from repo root, on the cluster):
    python -m scripts.analysis.estimate_ko_publication_size
"""

import os
import pickle
import subprocess
from pathlib import Path

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

NEUROPILS = ['AL', 'MB', 'PENP', 'VLNP', 'CX', 'GNG',
             'LX', 'SNP', 'INP', 'LH', 'OL', 'VMNP']
VARIANT = 'static'


def ko_dir_for(allTs_base, neuropil, variant=VARIANT):
    return os.path.join(allTs_base, f'meanZ_allTs_KO_{variant}_{neuropil}')


def du_sh(path):
    try:
        out = subprocess.run(['du', '-sh', path], capture_output=True, text=True, timeout=300)
        return out.stdout.split()[0] if out.returncode == 0 and out.stdout else 'n/a'
    except Exception as e:
        return f'error ({e})'


def required_frame_files(X_test, seq_len, seq_steps):
    """(recording, frame) pairs needed across every test sequence, deduped."""
    needed = set()
    for path in X_test:
        path = str(path)
        recording = path.split('/')[-2]
        try:
            start = int(path.split('_')[-1].split('.')[0])
        except ValueError:
            continue
        for f in range(start, start + (seq_len - 1) * seq_steps + 1, seq_steps):
            needed.add((recording, f))
    return needed


def main():
    config = load_config()
    logger = setup_logger(task_name='estimate_ko_publication_size', log_dir='logs/estimate_ko_publication_size')
    paths = config['paths']
    model_params = config['model']['parameters']

    if config['data']['task'] != 'State_Modality_Valence_16':
        logger.warning(
            f"config['data']['task'] is {config['data']['task']!r}, but sfig_ko_neuropils.py "
            "always uses State_Modality_Valence_16 -- run with TASK=State_Modality_Valence_16."
        )

    logger.info(f"Loading test set from: {paths['pickle_path']}")
    with open(paths['pickle_path'], 'rb') as f:
        _, _, X_test, _, _, _ = pickle.load(f)
    logger.info(f"Test set size: {len(X_test)} samples")

    seq_len, seq_steps = model_params['seq_len'], model_params['seq_steps']
    needed = required_frame_files(X_test, seq_len, seq_steps)
    logger.info(f"Unique (recording, frame) files required across all test sequences: {len(needed)}")
    logger.info(f"Unique recordings touched: {len(set(r for r, _ in needed))}")

    allTs_base = os.path.dirname(paths['allTs_path'].rstrip('/'))

    # Baseline (already on the drive as `intact`) -- how many required files actually exist there
    baseline_dir = paths['allTs_path']
    baseline_present = sum(
        1 for rec, f in needed
        if (Path(baseline_dir) / rec / f'{rec}_{f}.tiff').exists()
    )
    logger.info(f"Baseline ({baseline_dir}): {baseline_present}/{len(needed)} required files present")

    total_sample_bytes = 0
    total_sample_count = 0

    for neuropil in NEUROPILS:
        ko_dir = ko_dir_for(allTs_base, neuropil)
        exists = os.path.isdir(ko_dir)
        full_size = du_sh(ko_dir) if exists else 'MISSING'

        present = 0
        sample_bytes = 0
        sample_n = 0
        if exists:
            for rec, f in needed:
                fp = Path(ko_dir) / rec / f'{rec}_{f}.tiff'
                if fp.exists():
                    present += 1
                    if sample_n < 500:  # sample file sizes, don't stat all ~50k
                        try:
                            sample_bytes += fp.stat().st_size
                            sample_n += 1
                        except OSError:
                            pass

        est_size_mb = (sample_bytes / sample_n * present / 1e6) if sample_n else 0
        logger.info(
            f"{neuropil}: dir={'exists' if exists else 'MISSING'} full_size={full_size} "
            f"required_present={present}/{len(needed)} "
            f"est_test_only_size={est_size_mb:.0f}MB (sampled {sample_n} files)"
        )
        if sample_n:
            total_sample_bytes += sample_bytes
            total_sample_count += sample_n

    if total_sample_count:
        avg_file_bytes = total_sample_bytes / total_sample_count
        est_total_mb = avg_file_bytes * len(needed) * len(NEUROPILS) / 1e6
        logger.info(
            f"Rough estimate, ALL 12 neuropils, test-set-only scope: "
            f"~{est_total_mb / 1000:.1f}GB total "
            f"(avg file size {avg_file_bytes/1024:.0f}KB x {len(needed)} files x {len(NEUROPILS)} neuropils)"
        )


if __name__ == '__main__':
    main()
