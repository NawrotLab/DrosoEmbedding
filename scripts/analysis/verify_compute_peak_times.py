"""
Verification: does src/preprocessing/compute_peak_times.py reproduce the
existing IDs_logTs.pickle exactly, when run against the real raw NIfTI data?

Read-only -- reads raw NIfTIs and the existing peak-times pickle, writes
nothing, does not touch either. This is the trust-building step before that
module's output is used for anything: every existing model checkpoint
depends on the current IDs_logTs.pickle being unchanged.

Checks, per sampled recording already present in the existing pickle:
  1. detect_stimulus_epochs() returns exactly N_EPOCHS_EXPECTED *
     FRAMES_PER_EPOCH crossings (self-consistency: it IS in the existing
     pickle, so it must have passed this gate originally).
  2. compute_peak_times() reproduces the exact same list of peak frames
     already stored for that recording.

Usage (from repo root):
    python -m scripts.analysis.verify_compute_peak_times            # samples 10 recordings
    N_SAMPLE=30 python -m scripts.analysis.verify_compute_peak_times
"""

import os
import pickle
import random

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.preprocessing.compute_peak_times import (
    collect_niftis, detect_stimulus_epochs, compute_peak_times,
    N_EPOCHS_EXPECTED, FRAMES_PER_EPOCH,
)


def main():
    config = load_config()
    logger = setup_logger(task_name='verify_compute_peak_times', log_dir='logs/verify_compute_peak_times')
    paths = config['paths']

    logger.info(f"Loading existing peak-times pickle: {paths['peakIDs_Times_All']}")
    with open(paths['peakIDs_Times_All'], 'rb') as f:
        id_times_dict = pickle.load(f)

    path_data = f"{paths['data_root']}/Paul_LFM_Data"
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

    n_stage1_ok, n_stage2_ok = 0, 0
    for rec in sample:
        nifti_path = nifti_by_rec[rec]

        crossings = detect_stimulus_epochs(nifti_path)
        expected_n = N_EPOCHS_EXPECTED * FRAMES_PER_EPOCH
        if len(crossings) == expected_n:
            n_stage1_ok += 1
        else:
            logger.error(f'{rec}: stage 1 MISMATCH -- {len(crossings)} crossings, expected {expected_n}')

        recomputed = compute_peak_times(nifti_path)
        existing = list(id_times_dict[rec])
        if recomputed == existing:
            n_stage2_ok += 1
        else:
            only_existing = set(existing) - set(recomputed)
            only_recomputed = set(recomputed) - set(existing)
            logger.error(
                f'{rec}: stage 2 MISMATCH -- existing={len(existing)} recomputed={len(recomputed)} '
                f'only_in_existing(sample)={sorted(only_existing)[:10]} '
                f'only_in_recomputed(sample)={sorted(only_recomputed)[:10]}'
            )

    logger.info(f'Stage 1 (epoch QC gate)   : {n_stage1_ok}/{len(sample)} matched exactly')
    logger.info(f'Stage 2 (peak timepoints) : {n_stage2_ok}/{len(sample)} matched exactly')

    if n_stage1_ok == len(sample) and n_stage2_ok == len(sample):
        logger.info('MATCH: compute_peak_times.py reproduces the existing pickle exactly on this sample.')
        return 0
    logger.error('MISMATCH: do not trust compute_peak_times.py until this is understood.')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
