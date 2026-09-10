"""
Reproduces the "peak timepoint" selection that originally produced
IDs_logTs.pickle (paths['peakIDs_Times_All']) in an earlier, separate project
(MSc_DL4DrosoWBCI) -- ported here so the full pipeline from raw NIfTI to
training pickle lives in one reproducible place.

compute_peak_times(): baseline-subtracts and log10-transforms the recording
(this is where "logTs" comes from), then keeps any timepoint after
MIN_PEAK_FRAME where at least Z_CONSISTENCY_FRAC of z-slices show
above-(global-mean) log-activity -- i.e. broad, near-simultaneous activation
across most of the imaged volume.

The original project also had a stimulus-epoch QC gate (detect_stimulus_epochs
/ find_threshold_crossings, in Preprocessing/FindPeakTimes.py) that produced
an automatic first-pass recording-inclusion list. That list was then, per the
original Get_ID_logTs.py, used as a static filter -- not re-run live -- and
full-scale verification showed a live re-run of that gate does NOT agree with
which recordings ended up in the final IDs_logTs.pickle for ~62/240
recordings, meaning the original inclusion decision involved manual curation
on top of the automatic pass that can't be recovered from code alone.
Deliberately not ported here: this project has no pipeline for new/future
recordings, so the gate would have no use, and reproducing only the automatic
half of the decision would misrepresent it as more automatic than it was.
Recording inclusion for this project is instead always taken directly from
the existing IDs_logTs.pickle's keys.

get_baseline_frame() from the original code is not ported separately: it
computed the exact same bare-normalize -> mean-over-baseline-window ->
mean-over-z steps that load_and_normNIFTI(substract_Baseline=True) already
performs, so `load_and_normNIFTI(path, substract_Baseline=True, t_base=...)`
is used directly instead.

WARNING: this is a from-scratch re-implementation of code the user
originally wrote in a different project. Do NOT use its output to replace
the existing IDs_logTs.pickle, or anything built from it, without first
running scripts/analysis/verify_compute_peak_times.py against real data and
confirming an exact match -- every existing model checkpoint depends on that
file being unchanged.

Usage (from repo root):
    python -m src.preprocessing.compute_peak_times
    RECORDINGS_FILE=some.txt PEAK_TIMES_OUT=scratch.pickle python -m src.preprocessing.compute_peak_times
"""

import os
import pickle

import numpy as np
from tqdm import tqdm

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.utils.imgTools import load_and_normNIFTI

BASELINE_T         = [0, 250]      # baseline window, passed to load_and_normNIFTI
MIN_PEAK_FRAME     = 250           # peaks are only ever looked for after this frame
Z_CONSISTENCY_FRAC = 0.9           # fraction of z-slices that must be above threshold


def compute_peak_times(nifti_path):
    """The list of 'peak' (logTs) frame indices for one recording."""
    data = load_and_normNIFTI(nifti_path, substract_Baseline=True, t_base=BASELINE_T)
    log_data = np.log10(data + 1e-10)
    mean_activity_zt = log_data.mean(axis=(0, 1))  # (z, t)
    z_range, t_range = mean_activity_zt.shape
    threshold = log_data.mean()
    n_above = (mean_activity_zt >= threshold).sum(axis=0)  # per-t count across z
    z_needed = int(Z_CONSISTENCY_FRAC * z_range)  # int() truncation matches the original exactly
    return [t for t in range(t_range) if t >= MIN_PEAK_FRAME and n_above[t] >= z_needed]


def collect_niftis(path_data):
    """Same file-discovery logic as clean_dataset.py, for consistency."""
    niftis = []
    for root, _, files in os.walk(path_data):
        for nii in files[1:]:
            if not nii.startswith('._'):
                niftis.append(os.path.join(root, nii))
    return sorted(niftis)


def main():
    config = load_config()
    logger = setup_logger(task_name='compute_peak_times', log_dir='logs/slurm')

    path_data = f"{config['paths']['data_root']}/Paul_LFM_Data"
    niftis = collect_niftis(path_data)

    recordings_file = os.environ.get('RECORDINGS_FILE')
    recording_filter = None
    if recordings_file:
        with open(recordings_file) as f:
            recording_filter = {line.strip() for line in f if line.strip()}
        logger.info(f'Recording filter active: {len(recording_filter)} recordings from {recordings_file}')

    id_times_dict = {}
    for nifti in tqdm(niftis):
        file_name = os.path.basename(nifti).split('.')[0]
        if recording_filter and file_name not in recording_filter:
            continue

        id_times_dict[file_name] = compute_peak_times(nifti)
        logger.info(f'{file_name}: {len(id_times_dict[file_name])} peak timepoints')

    out_path = os.environ.get('PEAK_TIMES_OUT', 'results/preprocessing/IDs_logTs_recomputed.pickle')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(id_times_dict, f)
    logger.info(f'Wrote {len(id_times_dict)} recordings to {out_path}')
    logger.info('This is a re-derivation for verification purposes only -- it does '
                'NOT overwrite paths["peakIDs_Times_All"]. Run '
                'scripts/analysis/verify_compute_peak_times.py before trusting it for anything.')


if __name__ == '__main__':
    main()
