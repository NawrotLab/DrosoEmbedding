"""
Package the "intact" (unperturbed) condition's preprocessed frame data for
publication, one .tar per recording -- matches the existing
data/preprocessed_frames/intact.tars/<recording>.tar convention exactly, so
CustomDataset needs no changes to read it after extraction.

Scope: every recording EXCEPT the control condition
(exclude_controls -- same filter collect_image_paths() applies: a
recording is a control if its ID has 'C' at position -5). Controls are
never used by any of the 3 classification tasks, so publishing their
frames only bloats the published repo for no reproducibility benefit --
this was found and fixed manually once (81 of 245 recordings, ~8.9GB out
of intact.tars' 26.81GB); this script exists so a future regeneration
(e.g. if new recordings are added) doesn't reintroduce the same bloat.

Every frame present on disk is packaged per recording (not scoped to any
particular split or task) -- this is meant to support full
retrain/reevaluate reproducibility, not just one figure's test set (unlike
package_ko_test_frames.py, which is deliberately scoped to test-set-only
frames for exactly the one figure that needs KO data).

Read-only on the source frame directory -- only ever opens it for reading;
writes new .tar files to OUT_DIR, nothing else.

Usage (from repo root, on the cluster):
    python -m scripts.analysis.package_intact_frames
    OUT_DIR=/some/other/place python -m scripts.analysis.package_intact_frames
"""

import os
import shutil
import tarfile
from pathlib import Path

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


def log_disk_space(logger, label, path):
    """Report total/used/free for the filesystem containing `path`. Walks
    up to the nearest existing parent if `path` doesn't exist yet."""
    p = Path(path)
    while not p.exists():
        p = p.parent
    total, used, free = shutil.disk_usage(p)
    gb = 1024 ** 3
    logger.info(f"{label} ({p}): {free/gb:.1f}GB free / {total/gb:.1f}GB total "
                f"({used/gb:.1f}GB used)")
    return free


def is_control(recording: str) -> bool:
    """Same rule as collect_image_paths()'s exclude_controls filter."""
    return len(recording) >= 5 and recording[-5] == 'C'


def main():
    config = load_config()
    logger = setup_logger(task_name='package_intact_frames', log_dir='logs/package_intact_frames')
    paths = config['paths']

    source_dir = Path(paths['allTs_path'])
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source frame directory not found: {source_dir}")

    all_recordings = sorted(d.name for d in source_dir.iterdir() if d.is_dir())
    recordings = [r for r in all_recordings if not is_control(r)]
    n_control = len(all_recordings) - len(recordings)
    logger.info(f"Found {len(all_recordings)} recordings under {source_dir}, "
                f"excluding {n_control} control recordings -> packaging {len(recordings)}")

    out_dir = Path(os.environ.get('OUT_DIR', 'results/preprocessing/intact_frames'))
    out_dir.mkdir(parents=True, exist_ok=True)

    log_disk_space(logger, 'Source data location', source_dir)
    log_disk_space(logger, 'Output location', out_dir)

    n_written = 0
    for rec in recordings:
        rec_dir = source_dir / rec
        tar_path = out_dir / f'{rec}.tar'
        frames = sorted(rec_dir.glob(f'{rec}_*.tiff'))
        if not frames:
            logger.warning(f"{rec}: no frame files found, skipping")
            continue
        with tarfile.open(tar_path, 'w') as tar:
            for frame_path in frames:
                tar.add(frame_path, arcname=f'{rec}/{frame_path.name}')
        n_written += 1

    logger.info(f"Done. Wrote {n_written} tars to {out_dir}")
    logger.info("Read-only on source frame directory -- nothing there was modified.")


if __name__ == '__main__':
    main()
