"""
Diagnostic: baseline static fill visualisation
===============================================
For a sample of recordings, plots brain frames at several timepoints alongside
the proposed static fill image (mean of t=T_FILL_START..T_FILL_END).
Neuropil footprints are overlaid as coloured contours.

This script verifies whether the static fill image contains meaningful spatial
structure or is effectively zero (concern #1 with the baseline fill approach).

Usage (from repo root):
    python -m scripts.diagnostics.diag_static_fill_viz

Outputs:
    results/diagnostics/static_fill_viz/static_fill_viz.png
"""

import os
import pickle
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import tifffile
from scipy.ndimage import zoom

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.utils.neuropil_masks import NEUROPIL_NAMES, load_mask_3d, footprint_2d

# ── constants ─────────────────────────────────────────────────────────────────

T_FILL_START  = 100
T_FILL_END    = 250
TIMEPOINTS    = [50, 150, 250, 350, 500, 700]   # frames to show per recording
N_RECORDINGS  = 5
RANDOM_SEED   = 42

# neuropils to overlay as contours (name → colour)
OVERLAY = {
    'AL':   'cyan',
    'GNG':  'red',
    'OL':   'yellow',
    'VLNP': 'magenta',
}

OUT_DIR = 'results/diagnostics/static_fill_viz'


# ── helpers ───────────────────────────────────────────────────────────────────

def available_frames(rec_dir: Path) -> dict:
    """Return {frame_idx: tiff_path} for all TIFFs in a recording dir."""
    out = {}
    for p in rec_dir.glob('*.tiff'):
        try:
            t = int(p.stem.split('_')[-1])
            out[t] = p
        except ValueError:
            pass
    return out


def closest_frame(frame_map: dict, t: int):
    """Return (actual_t, path) for the frame closest to t."""
    if not frame_map:
        return None, None
    best = min(frame_map, key=lambda x: abs(x - t))
    return best, frame_map[best]


def load_frame(path) -> np.ndarray:
    return tifffile.imread(str(path)).astype(np.float32)


def resize_footprint(fp: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize boolean footprint to match TIFF dimensions if needed."""
    if fp.shape == target_shape:
        return fp
    zy = target_shape[0] / fp.shape[0]
    zx = target_shape[1] / fp.shape[1]
    return zoom(fp.astype(np.float32), (zy, zx), order=0) > 0.5


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng    = random.Random(RANDOM_SEED)
    config = load_config()
    logger = setup_logger(task_name='diag_static_fill_viz', log_dir='logs/diag_KO')

    allTs_dir = Path(config['paths']['allTs_path'])

    # sample recordings from training set
    with open(config['paths']['pickle_path'], 'rb') as f:
        X_train = pickle.load(f)[0]
    rec_names = list({Path(str(p)).parent.name for p in X_train
                      if (allTs_dir / Path(str(p)).parent.name).exists()})
    sample = rng.sample(rec_names, min(N_RECORDINGS, len(rec_names)))
    logger.info(f'Sampled recordings: {sample}')

    # column layout: one col per timepoint + one for the fill mean
    col_labels = [f't={t}' for t in TIMEPOINTS] + [f'Mean\nt={T_FILL_START}–{T_FILL_END}\n(fill)']
    n_cols = len(col_labels)
    n_rows = len(sample)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.2, n_rows * 2.2),
                             gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, rec_name in enumerate(sample):
        rec_nr    = rec_name.split('_')[-1]
        rec_dir   = allTs_dir / rec_name
        frame_map = available_frames(rec_dir)

        if not frame_map:
            logger.warning(f'No frames found for {rec_name}, skipping.')
            continue

        # load fill frames (t=T_FILL_START..T_FILL_END)
        fill_frames = [
            load_frame(path)
            for t, path in sorted(frame_map.items())
            if T_FILL_START <= t <= T_FILL_END
        ]
        fill_img = np.mean(fill_frames, axis=0) if fill_frames else None

        # determine shared vmax across all columns for this row
        sample_imgs = []
        for t in TIMEPOINTS:
            actual_t, path = closest_frame(frame_map, t)
            if path:
                sample_imgs.append(load_frame(path))
        if fill_img is not None:
            sample_imgs.append(fill_img)
        vmax = np.percentile(np.concatenate([i.ravel() for i in sample_imgs]), 99) if sample_imgs else 1.0
        vmax = max(vmax, 1e-6)

        # load neuropil footprints for overlay
        footprints = {}
        img_shape  = sample_imgs[0].shape if sample_imgs else None
        for name in OVERLAY:
            idx = int(np.where(NEUROPIL_NAMES == name)[0])
            m   = load_mask_3d(rec_nr, idx)
            if m is not None and img_shape is not None:
                fp = footprint_2d(m)
                footprints[name] = resize_footprint(fp, img_shape)

        def draw(ax, img, title, is_fill=False):
            ax.imshow(img, cmap='gray', vmin=0, vmax=vmax, aspect='auto')
            for name, fp in footprints.items():
                colour = OVERLAY[name]
                ax.contour(fp.astype(float), levels=[0.5],
                           colors=[colour], linewidths=0.6, alpha=0.8)
            if is_fill:
                for spine in ax.spines.values():
                    spine.set_edgecolor('lime')
                    spine.set_linewidth(2)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=7, pad=3,
                             color='lime' if is_fill else 'white',
                             fontweight='bold' if is_fill else 'normal')
            if col == 0:
                ax.set_ylabel(rec_name, fontsize=6, rotation=0,
                              labelpad=60, va='center')

        # plot timepoint columns
        for col, t in enumerate(TIMEPOINTS):
            actual_t, path = closest_frame(frame_map, t)
            ax = axes[row, col]
            if path:
                img = load_frame(path)
                draw(ax, img, f't={actual_t}')
            else:
                ax.set_facecolor('#111111')
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(f't={t}\n(missing)', fontsize=7, pad=3)

        # plot fill column (highlighted)
        ax_fill = axes[row, -1]
        if fill_img is not None:
            col = n_cols - 1
            draw(ax_fill, fill_img, col_labels[-1], is_fill=True)
            logger.info(f'  {rec_name}: fill mean={fill_img.mean():.5f}, '
                        f'max={fill_img.max():.5f}, '
                        f'frac_nonzero={(fill_img > 0).mean():.3f}')
        else:
            ax_fill.set_facecolor('#111111')
            ax_fill.set_xticks([]); ax_fill.set_yticks([])

    # legend for neuropil contours
    legend_patches = [
        plt.Line2D([0], [0], color=c, linewidth=1.5, label=n)
        for n, c in OVERLAY.items()
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=len(OVERLAY),
               fontsize=8, framealpha=0.3, facecolor='black', labelcolor='white',
               bbox_to_anchor=(0.5, 0.0))

    fig.patch.set_facecolor('black')
    fig.suptitle(
        f'Static fill diagnostic — fill = mean(t={T_FILL_START}..{T_FILL_END})  '
        f'[green border = fill column]',
        fontsize=10, color='white', y=1.01,
    )

    out_path = os.path.join(OUT_DIR, 'static_fill_viz.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    logger.info(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
