"""
Build a small cache of 2D neuropil-isolation masks, to replace the
per-sample "isolated" TIFFs that used to back Fig 4's
load_neuropil_masks() (src/visualization/visualize_interpretability.py).

Those TIFFs (meanZ_allTs_{neuropil}/...) were permanently deleted in the
2026-09-11 cluster cleanup. Rather than regenerating them, this derives the
same information directly from the anatomical
Neuropils12_Masks/Neuropils12Registered_{rec_nr}.nii files -- the same
source src/utils/neuropil_masks.py::load_mask_3d()/footprint_2d() already
use successfully for the (currently passing) KO figure.

Why this alignment is trustworthy (not a leap of faith): tracing
src/utils/imgTools.py::load_and_normNIFTI()'s own (pre-existing,
non-template) isolate_neuropil branch shows it applies the exact same
transpose(1, 0, 2) to the mask -- no extra rotation -- against data that
already went through rot90(k=3). That is the identical transform
load_mask_3d() applies. So this cache uses the same transform the original
(deleted) isolation pipeline already used to produce Fig 4 before.

Scope: only the test-set recordings for the currently-configured TASK
(same test-set-only scoping already used for the KO data, see
scripts/analysis/package_ko_test_frames.py) -- Fig 4 only ever needs
correctly-classified test-set samples for one task's model.

Output:
  - results/preprocessing/isolation_masks_2d.pickle
    {rec_nr: bool ndarray of shape (12, H, W)}, one entry per test-set
    recording that has a Neuropils12_Masks file. H, W are each
    recording's native (post-transpose) mask resolution -- resizing to a
    model's input size (e.g. 128x128) happens at consumption time in
    load_neuropil_masks(), same as the old per-sample TIFF path did via
    its own resize transform.
  - results/analysis/isolation_masks_examples/*.png
    Overlay sanity-check images (mask in red on top of a real frame) for
    a handful of recordings -- inspect these before trusting the cache.

Read-only w.r.t. all source data (frames, masks, split pickle). Only
writes to results/preprocessing/ and results/analysis/, never to any
published/production path.

Usage (on the cluster, needs real data + deps; TASK et al. same env vars
as other scripts reading the split pickle):
    python -m scripts.analysis.build_isolation_masks_cache [--n_examples 3]
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.utils.neuropil_masks import NEUROPIL_NAMES, footprint_2d, load_mask_3d, mask_path

CACHE_PATH = 'results/preprocessing/isolation_masks_2d.pickle'
EXAMPLES_DIR = 'results/analysis/isolation_masks_examples'


def recordings_from_test_set(pickle_path):
    """Return {rec_name: (rec_nr, one_example_frame_path)} from X_test, deduped."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    X_test = data[2]  # (X_train, X_val, X_test, Y_train, Y_val, Y_test)

    recs = {}
    for p in X_test:
        p = str(p)
        rec_name = os.path.basename(os.path.dirname(p))
        if rec_name not in recs:
            recs[rec_name] = (rec_name.split('_')[-1], p)
    return recs


def save_overlay_example(rec_name, frame_path, masks, out_dir, logger):
    frame = tifffile.imread(frame_path).astype(np.float32)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for name, m, ax in zip(NEUROPIL_NAMES, masks, axes.ravel()):
        if m.shape != frame.shape:
            logger.warning(f'{rec_name}/{name}: SHAPE MISMATCH mask {m.shape} vs frame {frame.shape}')
        ax.imshow(frame, cmap='gray')
        overlay = np.zeros((*m.shape, 4))
        overlay[m] = [1, 0, 0, 0.45]
        ax.imshow(overlay)
        ax.set_title(name)
        ax.axis('off')
    fig.suptitle(f'{rec_name}, frame {os.path.basename(frame_path)}')
    fig.tight_layout()
    out_path = os.path.join(out_dir, f'{rec_name}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f'Saved example overlay: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_examples', type=int, default=3,
                        help='Number of recordings to also save overlay PNGs for.')
    args = parser.parse_args()

    logger = setup_logger('build_isolation_masks_cache')
    config = load_config()

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    os.makedirs(EXAMPLES_DIR, exist_ok=True)

    recs = recordings_from_test_set(config['paths']['pickle_path'])
    logger.info(f'{len(recs)} unique test-set recordings found.')

    cache = {}
    n_missing_mask = 0
    example_count = 0

    for rec_name, (rec_nr, example_frame_path) in sorted(recs.items()):
        if not os.path.exists(mask_path(rec_nr)):
            n_missing_mask += 1
            continue

        footprints = []
        for idx in range(len(NEUROPIL_NAMES)):
            mask_3d = load_mask_3d(rec_nr, idx)
            footprints.append(footprint_2d(mask_3d))
        masks = np.stack(footprints, axis=0)  # (12, H, W) bool
        cache[rec_nr] = masks

        if example_count < args.n_examples:
            save_overlay_example(rec_name, example_frame_path, masks, EXAMPLES_DIR, logger)
            example_count += 1

    logger.info(f'Built masks for {len(cache)}/{len(recs)} recordings '
                f'({n_missing_mask} missing a Neuropils12_Masks file).')

    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)
    logger.info(f'Saved cache: {CACHE_PATH}')


if __name__ == '__main__':
    main()
