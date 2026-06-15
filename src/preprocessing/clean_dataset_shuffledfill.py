"""
Preprocessing: Neuropil KO with shuffled baseline fill
=======================================================
Alternative to clean_dataset_staticfill.py.

For each timepoint in the recording, the neuropil voxels are replaced with
values from a randomly sampled frame drawn from the baseline window
T_FILL_START..T_FILL_END (default 100..250).

Rationale: the *mean* of the baseline window in ΔF/F space is near-zero by
construction of the baseline subtraction.  Individual baseline frames contain
real spontaneous activity and spatial noise — they look like a brain at rest.
Shuffling randomly-drawn baseline frames into the neuropil at each timepoint
preserves realistic temporal variability and non-zero spatial structure while
completely removing the stimulus-driven response.

Output namespace:
    meanZ_allTs_KO_shuffled_{neuropil}

Usage (from repo root):
    NEUROPIL=AL TIMES=allTs python -m src.preprocessing.clean_dataset_shuffledfill

Env vars:
    NEUROPIL          — neuropil to knock out (required)
    TIMES             — logTs | allTs (default: allTs)
    METHOD_CH         — meanZ (default)
    RECORDINGS_FILE   — optional path to newline-separated recording names
    T_FILL_START      — first frame of baseline window (default: 100)
    T_FILL_END        — last frame of baseline window (default: 250)
    RANDOM_SEED       — integer seed for reproducibility (default: 42)
"""

import os
import sys
import pickle
import numpy as np
import tifffile
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.utils.imgTools import load_and_normNIFTI
from src.utils.neuropil_masks import NEUROPIL_NAMES, load_mask_3d


# ── fill function ─────────────────────────────────────────────────────────────

def apply_shuffledfill(rec_data: np.ndarray, mask_3d: np.ndarray,
                       t_start: int, t_end: int,
                       rng: np.random.RandomState) -> np.ndarray:
    """
    For each timepoint t, replace neuropil voxels with values from a randomly
    sampled frame drawn from [t_start, t_end). Each timepoint gets an
    independent random draw (fixed seed), preserving realistic temporal
    variability from spontaneous baseline activity. Returns a modified copy.
    """
    t_start  = max(0, t_start)
    t_end    = min(rec_data.shape[3], t_end)
    fill_ts  = np.arange(t_start, t_end)
    mask_idx = np.where(mask_3d)   # (y_idx, x_idx, z_idx)

    result = rec_data.copy()
    for t in range(result.shape[3]):
        t_fill    = rng.choice(fill_ts)
        fill_vals = rec_data[mask_idx[0], mask_idx[1], mask_idx[2], t_fill].astype(np.float32)
        result[mask_idx[0], mask_idx[1], mask_idx[2], t] = fill_vals

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    config = load_config()

    neuropil  = config['data']['preprocessing']['neuropil']
    timesID   = config['data']['preprocessing']['times']
    method_ch = config['data']['preprocessing']['method_ch']

    t_fill_start = int(os.environ.get('T_FILL_START', 100))
    t_fill_end   = int(os.environ.get('T_FILL_END',   250))
    random_seed  = int(os.environ.get('RANDOM_SEED',  42))

    neuropil_idx_arr = np.where(NEUROPIL_NAMES == neuropil)[0]
    if len(neuropil_idx_arr) == 0:
        raise ValueError(f'Unknown neuropil "{neuropil}". Choose from: {list(NEUROPIL_NAMES)}')
    neuropil_idx = int(neuropil_idx_arr[0])

    outputID    = f'{method_ch}_{timesID}_KO_shuffled_{neuropil}'
    path_data   = f'{config["paths"]["data_root"]}/Paul_LFM_Data'
    path_output = f'/localscratch/aabdel/imgs4DL/{outputID}'
    np_rng      = np.random.RandomState(random_seed)

    logger = setup_logger(task_name=f'shuffledfill_{neuropil}', log_dir='logs/slurm')
    logger.info(f'Neuropil     : {neuropil}  (index {neuropil_idx})')
    logger.info(f'Times        : {timesID}')
    logger.info(f'Fill window  : t={t_fill_start}..{t_fill_end} (shuffled per timepoint)')
    logger.info(f'Random seed  : {random_seed}')
    logger.info(f'Output       : {path_output}')

    os.makedirs(path_output, exist_ok=True)

    recordings_file  = os.environ.get('RECORDINGS_FILE')
    recording_filter = None
    if recordings_file:
        with open(recordings_file) as f:
            recording_filter = {line.strip() for line in f if line.strip()}
        logger.info(f'Recording filter: {len(recording_filter)} recordings from {recordings_file}')

    with open(config['paths']['peakIDs_Times_All'], 'rb') as f:
        id_times_dict = pickle.load(f)

    niftis = sorted([
        os.path.join(root, nii)
        for root, _, files in os.walk(path_data)
        for nii in files
        if not nii.startswith('._') and nii.endswith('.nii')
    ])
    logger.info(f'Found {len(niftis)} NIfTI files in {path_data}')

    for nifti in tqdm(niftis):
        file_name = nifti.split('/')[-1].split('.')[0]

        if recording_filter and file_name not in recording_filter:
            continue
        if file_name not in id_times_dict:
            continue

        rec_nr  = file_name.split('_')[-1]
        mask_3d = load_mask_3d(rec_nr, neuropil_idx)
        if mask_3d is None:
            logger.warning(f'No mask found for {file_name}, skipping.')
            continue

        rec_data = load_and_normNIFTI(
            nifti,
            substract_Baseline=True,
            t_base=[0, 250],
            save_baseframe=False,
            knockOutNeuropil=False,
            isolate_neuropil=False,
        )

        if rec_data.shape[:3] != mask_3d.shape:
            logger.warning(
                f'Shape mismatch for {file_name}: '
                f'data={rec_data.shape[:3]}, mask={mask_3d.shape} — skipping.'
            )
            continue

        # log fill stats from a single mid-baseline frame for inspection
        mid_t     = t_fill_start + (t_fill_end - t_fill_start) // 2
        mask_idx  = np.where(mask_3d)
        fill_sample = rec_data[mask_idx[0], mask_idx[1], mask_idx[2], mid_t]
        logger.info(
            f'Processing {file_name} | sample fill (t={mid_t}): '
            f'mean={fill_sample.mean():.5f}, '
            f'max={fill_sample.max():.5f}, '
            f'frac_nonzero={(fill_sample > 0).mean():.3f}'
        )

        rec_data = apply_shuffledfill(rec_data, mask_3d, t_fill_start, t_fill_end, np_rng)

        times = id_times_dict[file_name] if timesID == 'logTs' else np.arange(rec_data.shape[3])

        output_dir = os.path.join(path_output, file_name)
        os.makedirs(output_dir, exist_ok=True)
        for t in times:
            meanZ = np.nanmean(rec_data[:, :, :, t], axis=2)
            meanZ = np.nan_to_num(meanZ, nan=0.0)
            tifffile.imwrite(os.path.join(output_dir, f'{file_name}_{t}.tiff'), meanZ)


if __name__ == '__main__':
    main()
