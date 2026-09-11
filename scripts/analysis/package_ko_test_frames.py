"""
Package neuropil-KO frame data for publication, scoped to exactly what
sfig_ko_neuropils.py reads: the State_Modality_Valence_16 test set's
5-frame sequences (seq_len=5, seq_steps=10), for the 'static' KO variant,
across all 12 neuropils. See estimate_ko_publication_size.py for the
sizing rationale (~28GB total vs. ~312GB for full KO frame data).

For each neuropil, writes one .tar per recording actually touched by the
test set (not all 245 recordings -- only the ~163 that have test-set
sequences), containing only the required frame files for that recording
(not its full frame set) -- mirroring the existing
data/preprocessed_frames/intact/<recording>.tar convention exactly, so
CustomDataset needs no changes to read it after extraction.

Read-only on the source KO directories -- only ever opens them for
reading; writes new .tar files to OUT_DIR, nothing else. Reports free disk
space at both the source and output locations before writing anything,
and warns (but doesn't abort) if the estimated space needed looks larger
than what's free at the output location.

Usage (from repo root, on the cluster):
    TASK=State_Modality_Valence_16 python -m scripts.analysis.package_ko_test_frames
    OUT_DIR=/some/other/place python -m scripts.analysis.package_ko_test_frames
    NEUROPILS=AL,MB python -m scripts.analysis.package_ko_test_frames   # subset, for a test run
"""

import os
import pickle
import shutil
import tarfile
from pathlib import Path

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

ALL_NEUROPILS = ['AL', 'MB', 'PENP', 'VLNP', 'CX', 'GNG',
                  'LX', 'SNP', 'INP', 'LH', 'OL', 'VMNP']
VARIANT = 'static'

# From estimate_ko_publication_size.py's real measurement on the cluster
# (avg 88KB/file) -- used only to sanity-check free space before writing,
# not for anything load-bearing.
EST_BYTES_PER_FILE = 88 * 1024


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


def required_frames_by_recording(X_test, seq_len, seq_steps):
    """{recording: sorted[frame_numbers]} across every test sequence, deduped."""
    by_rec = {}
    for path in X_test:
        path = str(path)
        recording = path.split('/')[-2]
        try:
            start = int(path.split('_')[-1].split('.')[0])
        except ValueError:
            continue
        frames = by_rec.setdefault(recording, set())
        for f in range(start, start + (seq_len - 1) * seq_steps + 1, seq_steps):
            frames.add(f)
    return {rec: sorted(frames) for rec, frames in by_rec.items()}


def main():
    config = load_config()
    logger = setup_logger(task_name='package_ko_test_frames', log_dir='logs/package_ko_test_frames')
    paths = config['paths']
    model_params = config['model']['parameters']

    if config['data']['task'] != 'State_Modality_Valence_16':
        logger.warning(
            f"config['data']['task'] is {config['data']['task']!r}, but sfig_ko_neuropils.py "
            "always uses State_Modality_Valence_16 -- run with TASK=State_Modality_Valence_16."
        )

    with open(paths['pickle_path'], 'rb') as f:
        _, _, X_test, _, _, _ = pickle.load(f)

    seq_len, seq_steps = model_params['seq_len'], model_params['seq_steps']
    by_rec = required_frames_by_recording(X_test, seq_len, seq_steps)
    n_files = sum(len(v) for v in by_rec.values())
    logger.info(f"Test set: {len(X_test)} samples -> {len(by_rec)} recordings, {n_files} unique frame files needed")

    allTs_base = os.path.dirname(paths['allTs_path'].rstrip('/'))
    neuropils_env = os.environ.get('NEUROPILS')
    neuropils = neuropils_env.split(',') if neuropils_env else ALL_NEUROPILS
    out_dir = Path(os.environ.get('OUT_DIR', 'results/preprocessing/ko_test_frames'))

    log_disk_space(logger, 'Source data location', allTs_base)
    free_bytes = log_disk_space(logger, 'Output location', out_dir)
    est_needed = n_files * EST_BYTES_PER_FILE * len(neuropils)
    gb = 1024 ** 3
    logger.info(f"Estimated space needed for {len(neuropils)} neuropil(s): ~{est_needed/gb:.1f}GB")
    if est_needed > free_bytes:
        logger.warning(
            f"Estimated need (~{est_needed/gb:.1f}GB) exceeds free space at the output "
            f"location (~{free_bytes/gb:.1f}GB) -- consider running fewer neuropils at once "
            f"(NEUROPILS=...) or setting OUT_DIR to a location with more room."
        )

    for neuropil in neuropils:
        ko_dir = Path(allTs_base) / f'meanZ_allTs_KO_{VARIANT}_{neuropil}'
        if not ko_dir.is_dir():
            logger.error(f"{neuropil}: source dir not found, skipping: {ko_dir}")
            continue

        neuropil_out = out_dir / neuropil
        neuropil_out.mkdir(parents=True, exist_ok=True)

        n_written, n_missing = 0, 0
        for rec, frames in by_rec.items():
            tar_path = neuropil_out / f'{rec}.tar'
            missing_this_rec = 0
            with tarfile.open(tar_path, 'w') as tar:
                for frame in frames:
                    src = ko_dir / rec / f'{rec}_{frame}.tiff'
                    if not src.exists():
                        missing_this_rec += 1
                        continue
                    tar.add(src, arcname=f'{rec}/{rec}_{frame}.tiff')
            if missing_this_rec:
                logger.warning(f"{neuropil}/{rec}: {missing_this_rec}/{len(frames)} required files missing from source")
                n_missing += missing_this_rec
            n_written += 1

        logger.info(f"{neuropil}: wrote {n_written} tars to {neuropil_out} "
                    f"({'no missing files' if n_missing == 0 else f'{n_missing} files missing'})")

    logger.info(f"Done. Output under: {out_dir}")
    logger.info("Read-only on source KO directories -- nothing there was modified.")


if __name__ == '__main__':
    main()
