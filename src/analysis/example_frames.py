"""
Results files behind Fig 1 panel b and Fig S1 (the only data-derived, non-svg
panels of those two figures). Both pick a small, fixed set of example frames
via a seeded random choice, then draw only those frames -- so once the choice
is made, the figure needs only the selected pixels, not the full raw/
preprocessed dataset. See scripts/figures/run_figure_overview.py and
run_sfigure_exp_design.py for the selection logic (compute_results() in each).

Plain .npz (no pickle), so it loads in any numpy version; dtypes are stored
exactly, so a figure drawn from the file is identical to one drawn from the
in-memory results. rec_name/frame_idx are kept alongside the pixels purely as
provenance (which recording each example came from), never re-read from disk.
"""

import os

import numpy as np

RESULTS_VERSION = 1


# ── Fig 1 panel b ────────────────────────────────────────────────────────────

def save_fig1_panel_b(path, cond_names, rec_names, frames):
    """cond_names: list[str] (3, e.g. ['Odor','Taste','Combi_M']);
    rec_names: list[str] (3, the recording each condition's frames came from);
    frames: (3, 6, H, W) float array, the 6 selected meanZ timepoints per condition."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path, version=np.array(RESULTS_VERSION),
        cond_names=np.array(cond_names), rec_names=np.array(rec_names),
        frames=np.asarray(frames),
    )


def load_fig1_panel_b(path):
    """Inverse of save_fig1_panel_b: (cond_names, rec_names, frames)."""
    with np.load(path, allow_pickle=False) as z:
        _check_version(path, z)
        return z['cond_names'].tolist(), z['rec_names'].tolist(), z['frames']


# ── Fig S1 ────────────────────────────────────────────────────────────────

def save_s1_frames(path, row_labels, target_hw,
                    inc_frames, inc_norm, inc_rec_names, inc_frame_idx,
                    exc_frames, exc_norm, exc_rec_names, exc_frame_idx):
    """row_labels: list[str] (n_rows, the GROUPINGS labels);
    target_hw: (H, W), the shape every frame was resized to;
    inc_frames / exc_frames: (n_rows, n_inc_or_exc, H, W) float, resized, pre-normalisation;
    inc_norm / exc_norm: (n_rows, n_inc_or_exc, 2) [lo, hi] per frame;
    inc_rec_names / exc_rec_names: (n_rows, n_inc_or_exc) str, provenance only;
    inc_frame_idx / exc_frame_idx: (n_rows, n_inc_or_exc) int, provenance only."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path, version=np.array(RESULTS_VERSION),
        row_labels=np.array(row_labels), target_hw=np.array(target_hw, dtype=np.int64),
        inc_frames=np.asarray(inc_frames), inc_norm=np.asarray(inc_norm, dtype=np.float64),
        inc_rec_names=np.array(inc_rec_names), inc_frame_idx=np.asarray(inc_frame_idx, dtype=np.int64),
        exc_frames=np.asarray(exc_frames), exc_norm=np.asarray(exc_norm, dtype=np.float64),
        exc_rec_names=np.array(exc_rec_names), exc_frame_idx=np.asarray(exc_frame_idx, dtype=np.int64),
    )


def load_s1_frames(path):
    """Inverse of save_s1_frames: a dict with the same keyword names as its arguments
    (minus `path`), ready to pass back with **load_s1_frames(path)."""
    with np.load(path, allow_pickle=False) as z:
        _check_version(path, z)
        return dict(
            row_labels=z['row_labels'].tolist(), target_hw=tuple(int(v) for v in z['target_hw']),
            inc_frames=z['inc_frames'], inc_norm=z['inc_norm'],
            inc_rec_names=z['inc_rec_names'], inc_frame_idx=z['inc_frame_idx'],
            exc_frames=z['exc_frames'], exc_norm=z['exc_norm'],
            exc_rec_names=z['exc_rec_names'], exc_frame_idx=z['exc_frame_idx'],
        )


def _check_version(path, npz):
    version = int(npz['version'])
    if version != RESULTS_VERSION:
        raise ValueError(f'{path} has layout version {version}, expected {RESULTS_VERSION}')
