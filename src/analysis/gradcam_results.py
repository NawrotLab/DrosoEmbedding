"""
Results file behind Fig 4 (Grad-CAM neuropil importance).

Everything the figure plots -- the pooled attribution maps (panel b), and the
group / contrast tables (panels d, e) -- fits in one small .npz (well under
5 MB), so the figure can be drawn from it without the model, the test frames
or the neuropil masks. The heavy computation that produces it lives in
scripts/figures/run_figure_gradcam_neuropils.py (compute_results()).

A plain .npz (no pickle) so it loads in any numpy version; dtypes are stored
exactly, so a figure drawn from the file is identical to one drawn from the
in-memory results.
"""

import os

import numpy as np
import pandas as pd

RESULTS_VERSION = 1


def save_gradcam_results(path, pooled_cams, brain_shape, df_group, df_contrast):
    """pooled_cams: dict name -> (H, W) array (insertion order is kept);
    brain_shape: (H, W); df_group / df_contrast: DataFrames (rows: groups /
    contrasts, columns: neuropils)."""
    names = list(pooled_cams)
    arrays = {
        'version': np.array(RESULTS_VERSION),
        'pooled_names': np.array(names),
        'brain_shape': np.array(brain_shape, dtype=np.int64),
        'group_values': df_group.to_numpy(),
        'group_index': np.array(df_group.index.tolist()),
        'contrast_values': df_contrast.to_numpy(),
        'contrast_index': np.array(df_contrast.index.tolist()),
        'neuropils': np.array(df_group.columns.tolist()),
    }
    for i, name in enumerate(names):
        arrays[f'pooled_{i}'] = np.asarray(pooled_cams[name])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_gradcam_results(path):
    """Inverse of save_gradcam_results: (pooled_cams, brain_shape, df_group, df_contrast)."""
    with np.load(path, allow_pickle=False) as z:
        version = int(z['version'])
        if version != RESULTS_VERSION:
            raise ValueError(f'{path} has layout version {version}, expected {RESULTS_VERSION}')
        names = z['pooled_names'].tolist()
        pooled_cams = {name: z[f'pooled_{i}'] for i, name in enumerate(names)}
        brain_shape = tuple(int(v) for v in z['brain_shape'])
        neuropils = z['neuropils'].tolist()
        df_group = pd.DataFrame(z['group_values'], index=z['group_index'].tolist(), columns=neuropils)
        df_contrast = pd.DataFrame(z['contrast_values'], index=z['contrast_index'].tolist(), columns=neuropils)
    return pooled_cams, brain_shape, df_group, df_contrast
