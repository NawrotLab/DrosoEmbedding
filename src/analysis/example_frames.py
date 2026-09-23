"""
Results file behind Fig 1 panel b and Fig S1 (the only data-derived, non-svg
panels of those two figures) -- one shared file, example_frames.npz, since
both are small collections of "example frames" picked via a seeded random
choice and never recomputed at level 1. See scripts/figures/run_fig1_overview.py
and run_figS1_exp_design.py for the selection logic (compute_results() in each).

The two halves have different raw-data dependencies (Fig 1 needs
paths['raw_recordings'], S1 needs paths['allTs_path']), so someone may
recompute one without the other -- save_fig1_panel_b()/save_s1_frames() each
preserve the other half's keys if the file already exists, rather than
overwriting the whole file.

Plain .npz (no pickle), so it loads in any numpy version; dtypes are stored
exactly, so a figure drawn from the file is identical to one drawn from the
in-memory results. rec_name/frame_idx are kept alongside the pixels purely as
provenance (which recording each example came from), never re-read from disk.
"""

import os

import numpy as np

RESULTS_VERSION = 1

_FIG1_PREFIX = 'fig1_'
_S1_PREFIX = 's1_'


def _load_raw(path):
    """All arrays currently in the file, keyed as stored (with prefixes)."""
    if not os.path.exists(path):
        return {}
    with np.load(path, allow_pickle=False) as z:
        version = int(z['version']) if 'version' in z.files else None
        if version is not None and version != RESULTS_VERSION:
            raise ValueError(f'{path} has layout version {version}, expected {RESULTS_VERSION}')
        return {k: z[k] for k in z.files if k != 'version'}


def _save_raw(path, arrays):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, version=np.array(RESULTS_VERSION), **arrays)


def _strip_prefix(d, prefix):
    return {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}


def _drop_prefix_keys(d, prefix):
    return {k: v for k, v in d.items() if not k.startswith(prefix)}


# ── Fig 1 panel b ────────────────────────────────────────────────────────────

def save_fig1_panel_b(path, cond_names, rec_names, frames):
    """cond_names: list[str] (n_cond, e.g. ['Odor','Taste','Combi_M']);
    rec_names: list[str] (n_cond, the recording each condition's frames came from);
    frames: list of (6, H, W) float arrays, one per condition -- the 6 selected
    meanZ timepoints. Each condition is a *different* raw recording, so its H, W
    need not match the others' (stored as separate arrays, not one stacked
    tensor, for exactly that reason)."""
    assert len(cond_names) == len(rec_names) == len(frames)
    existing = _drop_prefix_keys(_load_raw(path), _FIG1_PREFIX)  # keep S1's half, if any
    mine = {'cond_names': np.array(cond_names), 'rec_names': np.array(rec_names)}
    for i, f in enumerate(frames):
        mine[f'frames_{i}'] = np.asarray(f)
    existing.update({_FIG1_PREFIX + k: v for k, v in mine.items()})
    _save_raw(path, existing)


def has_fig1_panel_b(path: str) -> bool:
    """True if path exists AND already has Fig 1 panel b's half saved --
    NOT the same as os.path.exists(path), since the file is shared with S1
    and may exist with only the other half present."""
    return any(k.startswith(_FIG1_PREFIX) for k in _load_raw(path))


def load_fig1_panel_b(path):
    """Inverse of save_fig1_panel_b: (cond_names, rec_names, frames) --
    frames: list of (6, H, W) arrays, one per condition, in cond_names order."""
    d = _strip_prefix(_load_raw(path), _FIG1_PREFIX)
    cond_names = d['cond_names'].tolist()
    frames = [d[f'frames_{i}'] for i in range(len(cond_names))]
    return cond_names, d['rec_names'].tolist(), frames


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
    existing = _drop_prefix_keys(_load_raw(path), _S1_PREFIX)  # keep Fig 1's half, if any
    mine = dict(
        row_labels=np.array(row_labels), target_hw=np.array(target_hw, dtype=np.int64),
        inc_frames=np.asarray(inc_frames), inc_norm=np.asarray(inc_norm, dtype=np.float64),
        inc_rec_names=np.array(inc_rec_names), inc_frame_idx=np.asarray(inc_frame_idx, dtype=np.int64),
        exc_frames=np.asarray(exc_frames), exc_norm=np.asarray(exc_norm, dtype=np.float64),
        exc_rec_names=np.array(exc_rec_names), exc_frame_idx=np.asarray(exc_frame_idx, dtype=np.int64),
    )
    existing.update({_S1_PREFIX + k: v for k, v in mine.items()})
    _save_raw(path, existing)


def has_s1_frames(path: str) -> bool:
    """True if path exists AND already has Fig S1's half saved -- NOT the
    same as os.path.exists(path), since the file is shared with Fig 1 panel b
    and may exist with only the other half present."""
    return any(k.startswith(_S1_PREFIX) for k in _load_raw(path))


def load_s1_frames(path):
    """Inverse of save_s1_frames: a dict with the same keyword names as its arguments
    (minus `path`), ready to pass back with **load_s1_frames(path)."""
    d = _strip_prefix(_load_raw(path), _S1_PREFIX)
    return dict(
        row_labels=d['row_labels'].tolist(), target_hw=tuple(int(v) for v in d['target_hw']),
        inc_frames=d['inc_frames'], inc_norm=d['inc_norm'],
        inc_rec_names=d['inc_rec_names'], inc_frame_idx=d['inc_frame_idx'],
        exc_frames=d['exc_frames'], exc_norm=d['exc_norm'],
        exc_rec_names=d['exc_rec_names'], exc_frame_idx=d['exc_frame_idx'],
    )
