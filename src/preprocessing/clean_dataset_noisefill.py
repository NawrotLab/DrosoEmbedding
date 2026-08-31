"""
Preprocessing: Neuropil KO with Gaussian noise-fill
====================================================
Alternative to clean_dataset.py where masked voxels are replaced with
Gaussian noise matched to training-set statistics of that neuropil's
pixels, rather than NaN/zero.

Rationale: NaN→0 creates a large black region for big neuropils (OL, VLNP),
producing out-of-distribution inputs for the model. Noise-fill keeps the
image in the training distribution while still removing the neuropil's
true activity signal.

Stats computation:
    For each neuropil, we sample training-set TIFF frames from meanZ_allTs,
    extract pixel values at the 2D footprint of the neuropil mask, and
    compute mean and std. Stats are cached to avoid recomputation.

Output namespace (completely separate from existing KO dirs):
    meanZ_allTs_KO_noisefill_{neuropil}
    meanZ_logTs_KO_noisefill_{neuropil}

Usage (from repo root):
    NEUROPIL=AL TIMES=allTs python -m src.preprocessing.clean_dataset_noisefill

Env vars (same as clean_dataset.py):
    NEUROPIL         — neuropil to knock out (required)
    TIMES            — logTs | allTs (default: allTs)
    METHOD_CH        — meanZ (default)
    RECORDINGS_FILE  — optional path to newline-separated recording names
"""

import os
import sys
import json
import pickle
import random
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.utils.imgTools import load_and_normNIFTI
from src.utils.neuropil_masks import NEUROPIL_NAMES, load_mask_3d, footprint_2d, mask_path

# ── constants ─────────────────────────────────────────────────────────────────

STATS_CACHE    = 'results/preprocessing/noisefill_stats.json'
N_STATS_RECS   = 30   # recordings sampled from training set for stat computation
N_STATS_FRAMES = 10   # frames per recording
RANDOM_SEED    = 42


# ── noise statistics ──────────────────────────────────────────────────────────

def compute_or_load_stats(config: dict, neuropil: str, neuropil_idx: int,
                           rng: random.Random, logger) -> tuple:
    """
    Return (mean, std) of baseline pixel values outside ALL neuropil masks,
    sampled from the training set. Stats are background-tissue statistics,
    independent of which neuropil is being knocked out. Cached under key
    'background' in STATS_CACHE.
    """
    os.makedirs(os.path.dirname(STATS_CACHE), exist_ok=True)

    cache = {}
    if os.path.exists(STATS_CACHE):
        with open(STATS_CACHE) as f:
            cache = json.load(f)
    if 'background' in cache:
        mean, std = cache['background']['mean'], cache['background']['std']
        logger.info(f'Loaded cached background stats: mean={mean:.5f}, std={std:.5f}')
        return mean, std

    logger.info('Computing background noise stats (outside all neuropil masks) from training set...')

    with open(config['paths']['pickle_path'], 'rb') as f:
        X_train = pickle.load(f)[0]

    allTs_dir = Path(config['paths']['allTs_path'])

    all_rec_names = list({Path(str(p)).parent.name for p in X_train
                          if (allTs_dir / Path(str(p)).parent.name).exists()})
    sample_recs   = rng.sample(all_rec_names, min(N_STATS_RECS, len(all_rec_names)))

    pixel_values = []
    for rec_name in sample_recs:
        rec_nr = rec_name.split('_')[-1]

        # union footprint of all 12 neuropils
        union_fp = None
        for idx in range(len(NEUROPIL_NAMES)):
            m = load_mask_3d(rec_nr, idx)
            if m is None:
                continue
            fp_i = footprint_2d(m)
            union_fp = fp_i if union_fp is None else (union_fp | fp_i)

        if union_fp is None:
            continue
        background = ~union_fp   # pixels outside all neuropils

        tiffs = sorted((allTs_dir / rec_name).glob('*.tiff'))
        if not tiffs:
            continue
        sample_tiffs = rng.sample(tiffs, min(N_STATS_FRAMES, len(tiffs)))

        for tp in sample_tiffs:
            img = tifffile.imread(str(tp)).astype(np.float32)
            if img.shape != background.shape:
                continue
            vals = img[background]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            pixel_values.extend(vals.tolist())

    if not pixel_values:
        logger.warning('No background pixel values found — using fallback stats (0, 0.005).')
        return 0.0, 0.005

    arr  = np.array(pixel_values, dtype=np.float32)
    mean = float(arr.mean())
    std  = float(arr.std())
    logger.info(f'Background stats: n={len(arr):,}, mean={mean:.5f}, std={std:.5f}')

    cache['background'] = {'mean': mean, 'std': std}
    with open(STATS_CACHE, 'w') as f:
        json.dump(cache, f, indent=2)

    return mean, std


# ── noise-fill function ───────────────────────────────────────────────────────

def apply_noisefill(rec_data: np.ndarray, mask_3d: np.ndarray,
                    noise_mean: float, noise_std: float,
                    rng_state: np.random.RandomState) -> np.ndarray:
    """
    Replace voxels within mask_3d with i.i.d. Gaussian noise N(mean, std),
    clipped to [0, inf).  Returns a modified copy.
    """
    result      = rec_data.copy()
    mask_idx    = np.where(mask_3d)          # (y_idx, x_idx, z_idx) arrays
    n_masked    = len(mask_idx[0])

    for t in range(result.shape[3]):
        noise = rng_state.normal(noise_mean, noise_std, size=n_masked).astype(np.float32)
        noise = np.clip(noise, 0.0, None)
        result[mask_idx[0], mask_idx[1], mask_idx[2], t] = noise

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    config = load_config()
    rng    = random.Random(RANDOM_SEED)
    np_rng = np.random.RandomState(RANDOM_SEED)

    neuropil  = config['data']['preprocessing']['neuropil']
    timesID   = config['data']['preprocessing']['times']
    method_ch = config['data']['preprocessing']['method_ch']

    neuropil_idx_arr = np.where(NEUROPIL_NAMES == neuropil)[0]
    if len(neuropil_idx_arr) == 0:
        raise ValueError(f'Unknown neuropil "{neuropil}". Choose from: {list(NEUROPIL_NAMES)}')
    neuropil_idx = int(neuropil_idx_arr[0])

    outputID    = f'{method_ch}_{timesID}_KO_noisefill_{neuropil}'
    path_data   = f'{config["paths"]["data_root"]}/Paul_LFM_Data'
    path_output = os.path.join(config['paths']['local_scratch_dir'], 'imgs4DL', outputID)

    logger = setup_logger(task_name=f'noisefill_{neuropil}', log_dir='logs/slurm')
    logger.info(f'Neuropil  : {neuropil}  (index {neuropil_idx})')
    logger.info(f'Times     : {timesID}')
    logger.info(f'Output    : {path_output}')

    os.makedirs(path_output, exist_ok=True)

    # optional recording filter
    recordings_file  = os.environ.get('RECORDINGS_FILE')
    recording_filter = None
    if recordings_file:
        with open(recordings_file) as f:
            recording_filter = {line.strip() for line in f if line.strip()}
        logger.info(f'Recording filter: {len(recording_filter)} recordings from {recordings_file}')

    # compute / load noise statistics
    noise_mean, noise_std = compute_or_load_stats(
        config, neuropil, neuropil_idx, rng, logger
    )

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

        logger.info(f'Processing {file_name}')

        # load + normalise (no KO — we handle masking ourselves)
        rec_data = load_and_normNIFTI(
            nifti,
            substract_Baseline=True,
            t_base=[0, 250],
            save_baseframe=False,
            knockOutNeuropil=False,
            isolate_neuropil=False,
        )

        # verify spatial dimensions match before applying mask
        if rec_data.shape[:3] != mask_3d.shape:
            logger.warning(
                f'Shape mismatch for {file_name}: '
                f'data={rec_data.shape[:3]}, mask={mask_3d.shape} — skipping.'
            )
            continue

        # replace neuropil voxels with Gaussian noise
        rec_data = apply_noisefill(rec_data, mask_3d, noise_mean, noise_std, np_rng)

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
