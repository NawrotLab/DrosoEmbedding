"""
One-shot migration: convert an existing results/ folder from the pre-merge
layout (separate fig1_panel_b.npz + figS1_frames.npz, and a ko_static/
raw_delta_stack.npy + neuropil_sizes_2d.json pair) into the merged layout
(example_frames.npz, ko_static.npz) -- see:
  - commit "Merge Fig S4's stack + neuropil sizes into one results file"
  - commit "Merge Fig 1 panel b + Fig S1 results into one example_frames.npz"

Pure data reshuffling, no recomputation: reads whatever old-format files are
present and writes the merged equivalents. Safe to run more than once
(skips anything already in the new format, never overwrites without asking).

Usage (from repo root, needs only numpy/pandas -- no GPU, no torch):
    python -m scripts.analysis.migrate_results_layout
"""

import json
import os

import numpy as np

from src.utils.config_loader import load_config


def migrate_fig1_s1(results_dir, log):
    from src.analysis.example_frames import (
        has_fig1_panel_b, has_s1_frames, save_fig1_panel_b, save_s1_frames,
    )

    merged_path = os.path.join(results_dir, 'example_frames.npz')
    old_fig1 = os.path.join(results_dir, 'fig1_panel_b.npz')
    old_s1 = os.path.join(results_dir, 'figS1_frames.npz')

    if os.path.exists(old_fig1) and not has_fig1_panel_b(merged_path):
        with np.load(old_fig1, allow_pickle=False) as z:
            cond_names = z['cond_names'].tolist()
            rec_names = z['rec_names'].tolist()
            frames = [z[f'frames_{i}'] for i in range(len(cond_names))]
        save_fig1_panel_b(merged_path, cond_names, rec_names, frames)
        log(f'migrated {old_fig1} -> {merged_path} (fig1_ half)')
    elif has_fig1_panel_b(merged_path):
        log(f'{merged_path} already has its fig1_ half, skipping')
    else:
        log(f'no {old_fig1} found, nothing to migrate for Fig 1')

    if os.path.exists(old_s1) and not has_s1_frames(merged_path):
        with np.load(old_s1, allow_pickle=False) as z:
            kwargs = {k: z[k] for k in z.files}
        save_s1_frames(
            merged_path,
            row_labels=kwargs['row_labels'].tolist(),
            target_hw=tuple(int(v) for v in kwargs['target_hw']),
            inc_frames=kwargs['inc_frames'], inc_norm=kwargs['inc_norm'],
            inc_rec_names=kwargs['inc_rec_names'], inc_frame_idx=kwargs['inc_frame_idx'],
            exc_frames=kwargs['exc_frames'], exc_norm=kwargs['exc_norm'],
            exc_rec_names=kwargs['exc_rec_names'], exc_frame_idx=kwargs['exc_frame_idx'],
        )
        log(f'migrated {old_s1} -> {merged_path} (s1_ half)')
    elif has_s1_frames(merged_path):
        log(f'{merged_path} already has its s1_ half, skipping')
    else:
        log(f'no {old_s1} found, nothing to migrate for S1')


def migrate_ko(results_dir, log):
    from src.analysis.ko_permutation import NEUROPILS, save_ko_results

    merged_path = os.path.join(results_dir, 'ko_static.npz')
    old_stack = os.path.join(results_dir, 'ko_static', 'raw_delta_stack.npy')
    old_sizes = os.path.join(results_dir, 'neuropil_sizes_2d.json')

    if os.path.exists(merged_path):
        log(f'{merged_path} already exists, skipping')
        return
    if not (os.path.exists(old_stack) and os.path.exists(old_sizes)):
        log(f'no old-format S4 source found ({old_stack} / {old_sizes}), nothing to migrate')
        return

    stack = np.load(old_stack)
    sizes_json = json.load(open(old_sizes))
    sizes = np.array([sizes_json[n] for n in NEUROPILS])
    save_ko_results(merged_path, stack, sizes, NEUROPILS)
    log(f'migrated {old_stack} + {old_sizes} -> {merged_path} '
        f'(stack shape {stack.shape})')


def main():
    paths = load_config()['paths']
    results_dir = paths['results_dir']

    def log(msg):
        print(msg)

    log(f'results_dir: {results_dir}')
    migrate_fig1_s1(results_dir, log)
    migrate_ko(results_dir, log)
    log('done -- old-format files were NOT deleted, remove them by hand once you\'ve '
        'confirmed the merged files work (re-run the figure scripts and diff the PNGs).')


if __name__ == '__main__':
    main()
