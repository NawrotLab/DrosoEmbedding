"""
Verification: does the NEW collect_image_paths() (src/preprocessing/
XY_WithinAnimalSplitting.py), which looks anchor frames up in id_times_dict
instead of walking a physical logTs directory, reproduce the same
(recording, frame) universe as the existing train/val/test pickle?

Read-only -- calls the real collect_image_paths() function against your
existing meanZ_allTs directory, but writes no pickle and touches nothing.

Usage (from repo root):
    python -m scripts.analysis.verify_collect_image_paths
"""

import pickle

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.preprocessing.XY_WithinAnimalSplitting import collect_image_paths


def parse_rec_frame(path: str):
    recording = path.split('/')[-2]
    frame_idx = int(path.split('_')[-1].split('.')[0])
    return recording, frame_idx


def main():
    config = load_config()
    logger = setup_logger(task_name='verify_collect_image_paths', log_dir='logs/verify_collect_image_paths')
    paths = config['paths']
    args = config['data']['preprocessing']

    logger.info(f"Task: {config['data']['task']}")
    logger.info(f"Loading existing pickle: {paths['pickle_path']}")
    with open(paths['pickle_path'], 'rb') as f:
        X_train, X_val, X_test, Y_train, Y_val, Y_test = pickle.load(f)
    existing = {parse_rec_frame(str(p)) for p in (X_train + X_val + X_test)}
    logger.info(f"Existing pickle: {len(existing)} (recording, frame) samples")

    logger.info(f"Loading id_times_dict: {paths['peakIDs_Times_All']}")
    with open(paths['peakIDs_Times_All'], 'rb') as f:
        id_times_dict = pickle.load(f)

    processed_data_path = f'{paths["imgs4DL"]}/{args["method_ch"]}_allTs'
    logger.info(f"Calling collect_image_paths() against: {processed_data_path}")
    rec_paths_dict = collect_image_paths(
        processed_data_path, id_times_dict, 'logTs',
        args['exclude_controls'], logger,
    )
    reconstructed = {
        parse_rec_frame(p) for paths_list in rec_paths_dict.values() for p in paths_list
    }
    logger.info(f"collect_image_paths() output: {len(reconstructed)} (recording, frame) samples")

    only_in_existing = existing - reconstructed
    only_in_reconstructed = reconstructed - existing
    logger.info(f"In existing pickle but NOT reproduced: {len(only_in_existing)}")
    logger.info(f"Reproduced but NOT in existing pickle: {len(only_in_reconstructed)}")
    if only_in_existing:
        logger.warning(f"Sample: {sorted(only_in_existing)[:10]}")
    if only_in_reconstructed:
        logger.warning(f"Sample: {sorted(only_in_reconstructed)[:10]}")

    if not only_in_existing and not only_in_reconstructed:
        logger.info("MATCH: collect_image_paths() reproduces the existing pickle's sample universe exactly.")
        return 0
    logger.error("MISMATCH: collect_image_paths() does NOT reproduce the existing pickle. Do not use it yet.")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
