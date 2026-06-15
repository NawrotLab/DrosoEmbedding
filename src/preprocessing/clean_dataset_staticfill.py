"""
Preprocessing: Neuropil KO with shuffled baseline fill
=======================================================
Alternative to clean_dataset.py and clean_dataset_noisefill.py.

For each timepoint in the recording, the neuropil voxels are replaced with
the corresponding voxel values from a randomly sampled frame drawn from the
baseline window T_FILL_START..T_FILL_END (default 100..250).

Rationale: the mean of the baseline window in ΔF/F space is near-zero by
construction of the baseline subtraction.  Individual baseline frames,
however, contain real spontaneous activity and spatial noise — they look like
a brain at rest.  By shuffling randomly-drawn baseline frames into the
neuropil at each timepoint, we preserve realistic temporal variability and
non-zero spatial structure while completely removing the stimulus-driven
response of the neuropil.

Output namespace:
    meanZ_allTs_KO_shuffled_{neuropil}

Usage (from repo root):
    NEUROPIL=AL TIMES=allTs python -m src.preprocessing.clean_dataset_staticfill

Env vars (same as other preprocessing scripts):
    NEUROPIL          — neuropil to knock out (required)
    TIMES             — logTs | allTs (default: allTs)
    METHOD_CH         — meanZ (default)
    RECORDINGS_FILE   — optional path to newline-separated recording names
    T_FILL_START      — first frame of baseline window (default: 100)
    T_FILL_END        — last frame of baseline window (default: 250)
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

def apply_staticfill(rec_data: np.ndarray, mask_3d: np.ndarray,
                     t_start: int, t_end: int) -> np.ndarray:
    """
    Replace neuropil voxels with the mean baseline activity over t_start..t_end.
    The fill is computed per-recording and held constant across all timepoints.
    Returns a modified copy.
    """
    t_start = max(0, t_start)
    t_end   = min(rec_data.shape[3], t_end)

    fill_3d  = np.mean(rec_data[:, :, :, t_start:t_end], axis=3)  # (y, x, z)
    mask_idx = np.where(mask_3d)                                    # (y_idx, x_idx, z_idx)
    fill_vals = fill_3d[mask_idx].astype(np.float32)

    result = rec_data.copy()
    for t in range(result.shape[3]):
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

    neuropil_idx_arr = np.where(NEUROPIL_NAMES == neuropil)[0]
    if len(neuropil_idx_arr) == 0:
        raise ValueError(f'Unknown neuropil "{neuropil}". Choose from: {list(NEUROPIL_NAMES)}')
    neuropil_idx = int(neuropil_idx_arr[0])

    outputID    = f'{method_ch}_{timesID}_KO_static_{neuropil}'
    path_data   = f'{config["paths"]["data_root"]}/Paul_LFM_Data'
    path_output = f'/localscratch/aabdel/imgs4DL/{outputID}'

    logger = setup_logger(task_name=f'staticfill_{neuropil}', log_dir='logs/slurm')
    logger.info(f'Neuropil     : {neuropil}  (index {neuropil_idx})')
    logger.info(f'Times        : {timesID}')
    logger.info(f'Fill window  : t={t_fill_start}..{t_fill_end}')
    logger.info(f'Output       : {path_output}')

    os.makedirs(path_output, exist_ok=True)

    # optional recording filter
    recordings_file  = os.environ.get('RECORDINGS_FILE')
    recording_filter = None
    if recordings_file:
        with open(recordings_file) as f:
            recording_filter = {line.strip() for line in f if line.strip()}
        logger.info(f'Recording filter: {len(recording_filter)} recordings from {recordings_file}')

    # timepoint index (for logTs mode)
    with open(config['paths']['peakIDs_Times_All'], 'rb') as f:
        id_times_dict = pickle.load(f)

    # collect NIfTI files
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

        # load + normalise with standard baseline subtraction
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

        # compute and log fill stats for transparency
        fill_3d   = np.mean(rec_data[:, :, :, t_fill_start:t_fill_end], axis=3)
        fill_vals = fill_3d[mask_3d]
        logger.info(
            f'Processing {file_name} | fill: '
            f'mean={fill_vals.mean():.5f}, '
            f'max={fill_vals.max():.5f}, '
            f'frac_nonzero={(fill_vals > 0).mean():.3f}'
        )

        # apply static fill
        rec_data = apply_staticfill(rec_data, mask_3d, t_fill_start, t_fill_end)

        # determine timepoints to save
        times = id_times_dict[file_name] if timesID == 'logTs' else np.arange(rec_data.shape[3])

        # mean-Z project and write TIFFs
        output_dir = os.path.join(path_output, file_name)
        os.makedirs(output_dir, exist_ok=True)
        for t in times:
            meanZ = np.nanmean(rec_data[:, :, :, t], axis=2)
            meanZ = np.nan_to_num(meanZ, nan=0.0)
            tifffile.imwrite(os.path.join(output_dir, f'{file_name}_{t}.tiff'), meanZ)


if __name__ == '__main__':
    main()
