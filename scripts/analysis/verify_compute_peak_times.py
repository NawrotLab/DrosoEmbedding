"""
Verification: does src/preprocessing/compute_peak_times.py reproduce the
existing IDs_logTs.pickle exactly, when run against the real raw NIfTI data?

Read-only -- reads raw NIfTIs and the existing peak-times pickle, writes
nothing, does not touch either. This is the trust-building step before that
module's output is used for anything: every existing model checkpoint
depends on the current IDs_logTs.pickle being unchanged.

Check, per sampled recording already present in the existing pickle:
  compute_peak_times() reproduces the exact same list of peak frames already
  stored for that recording.

Also writes a per-recording CSV of existing vs. recomputed frame counts (see
RESULTS_OUT below) -- a standing artifact for eyeballing/matching against the
existing pickle's counts as an extra sanity check, independent of the
exact-match logic above.

(There used to also be a Stage 1 stimulus-epoch QC gate check here, mirroring
detect_stimulus_epochs() from the original project. That gate was discarded
from compute_peak_times.py -- see its module docstring -- since full-scale
verification showed a live re-run of it disagrees with the recordings actually
present in IDs_logTs.pickle for ~62/240 cases, meaning the original inclusion
decision involved manual curation the gate alone can't reproduce, and this
project has no use for it since recording inclusion is always taken directly
from the existing pickle's keys.)

Usage (from repo root):
    python -m scripts.analysis.verify_compute_peak_times            # samples 10 recordings
    N_SAMPLE=30 python -m scripts.analysis.verify_compute_peak_times
    RESULTS_OUT=scratch.csv python -m scripts.analysis.verify_compute_peak_times
"""

import csv
import os
import pickle
import random

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.preprocessing.compute_peak_times import collect_niftis, compute_peak_times


def main():
    config = load_config()
    logger = setup_logger(task_name='verify_compute_peak_times', log_dir='logs/verify_compute_peak_times')
    paths = config['paths']

    logger.info(f"Loading existing peak-times pickle: {paths['peakIDs_Times_All']}")
    with open(paths['peakIDs_Times_All'], 'rb') as f:
        id_times_dict = pickle.load(f)

    path_data = paths['raw_recordings']
    logger.info(f'Collecting NIfTI files under: {path_data}')
    niftis = collect_niftis(path_data)
    nifti_by_rec = {os.path.basename(p).split('.')[0]: p for p in niftis}

    n_sample = int(os.environ.get('N_SAMPLE', 10))
    candidates = [r for r in id_times_dict if r in nifti_by_rec]
    if len(candidates) < len(id_times_dict):
        logger.warning(f'{len(id_times_dict) - len(candidates)} recordings in the pickle '
                        f'have no matching NIfTI file under {path_data}')

    random.seed(777)
    sample = random.sample(candidates, min(n_sample, len(candidates)))
    logger.info(f'Verifying {len(sample)} of {len(candidates)} matchable recordings')

    n_ok = 0
    rows = []
    for rec in sample:
        nifti_path = nifti_by_rec[rec]

        recomputed = compute_peak_times(nifti_path)
        existing = list(id_times_dict[rec])
        match = recomputed == existing
        if match:
            n_ok += 1
        else:
            only_existing = set(existing) - set(recomputed)
            only_recomputed = set(recomputed) - set(existing)
            logger.error(
                f'{rec}: MISMATCH -- existing={len(existing)} recomputed={len(recomputed)} '
                f'only_in_existing(sample)={sorted(only_existing)[:10]} '
                f'only_in_recomputed(sample)={sorted(only_recomputed)[:10]}'
            )
        rows.append((rec, len(existing), len(recomputed), match))

    results_out = os.environ.get('RESULTS_OUT', 'results/analysis/verify_compute_peak_times_counts.csv')
    os.makedirs(os.path.dirname(results_out), exist_ok=True)
    with open(results_out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['recording', 'existing_count', 'recomputed_count', 'match'])
        writer.writerows(rows)
    logger.info(f'Wrote per-recording frame counts to {results_out}')

    logger.info(f'Peak timepoints matched exactly: {n_ok}/{len(sample)}')

    if n_ok == len(sample):
        logger.info('MATCH: compute_peak_times.py reproduces the existing pickle exactly on this sample.')
        return 0
    logger.error('MISMATCH: do not trust compute_peak_times.py until this is understood.')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
