"""
Verification: does id_times_dict (the peak-times file, paths['peakIDs_Times_All'])
fully reproduce the (recording, frame, label) universe of the existing
train/val/test pickle (paths['pickle_path'])?

Read-only. Loads the existing pickle and id_times_dict, but writes nothing and
does not touch either file. Run this BEFORE removing the physical logTs
directory / the 'logTs' branch from clean_dataset.py and
XY_WithinAnimalSplitting.py -- if it reports a full match, rebuilding the
anchor list directly from id_times_dict (instead of walking a materialized
logTs directory) is a safe, faithful replacement. If it reports any
mismatch, do NOT remove anything until the mismatch is understood.

Usage (from repo root):
    python -m scripts.analysis.verify_logts_removal
"""

import pickle

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


def parse_rec_frame(path: str):
    """Extract (recording, frame_idx) from a logTs/allTs-style anchor path
    (e.g. '.../meanZ_logTs/SOP_1/SOP_1_450.tiff' -> ('SOP_1', 450))."""
    recording = path.split('/')[-2]
    frame_idx = int(path.split('_')[-1].split('.')[0])
    return recording, frame_idx


def existing_triples_by_split(pickle_path):
    """(recording, frame_idx, label) triples from the existing pickle, per split."""
    with open(pickle_path, 'rb') as f:
        X_train, X_val, X_test, Y_train, Y_val, Y_test = pickle.load(f)

    splits = {}
    for name, X, Y in [('train', X_train, Y_train), ('val', X_val, Y_val), ('test', X_test, Y_test)]:
        triples = set()
        for path, label in zip(X, Y):
            rec, frame = parse_rec_frame(str(path))
            triples.add((rec, frame, label))
        splits[name] = triples
    return splits


def reconstruct_from_id_times(id_times_dict, rec_to_label):
    """(recording, frame_idx, label) universe implied by id_times_dict, restricted
    to the recordings the existing pickle actually used (and their known labels)."""
    triples = set()
    for rec, label in rec_to_label.items():
        if rec not in id_times_dict:
            continue
        for frame in id_times_dict[rec]:
            triples.add((rec, int(frame), label))
    return triples


def main():
    config = load_config()
    logger = setup_logger(task_name='verify_logts_removal', log_dir='logs/verify_logts_removal')
    paths = config['paths']

    logger.info(f"Task: {config['data']['task']}")
    logger.info(f"Loading existing pickle: {paths['pickle_path']}")
    existing = existing_triples_by_split(paths['pickle_path'])

    all_existing = existing['train'] | existing['val'] | existing['test']
    logger.info(
        f"Existing pickle: {len(all_existing)} total (recording, frame, label) samples "
        f"({len(existing['train'])} train / {len(existing['val'])} val / {len(existing['test'])} test)"
    )

    # recording -> label, taken directly from the existing pickle -- this
    # verification checks frame-universe equivalence, not the labeling logic,
    # so it reuses whatever labels the existing pickle already assigned.
    rec_to_label = {}
    inconsistent_recs = set()
    for rec, frame, label in all_existing:
        if rec in rec_to_label and rec_to_label[rec] != label:
            inconsistent_recs.add(rec)
        rec_to_label[rec] = label
    if inconsistent_recs:
        logger.warning(
            f"{len(inconsistent_recs)} recordings have inconsistent labels across "
            f"samples within the existing pickle itself (unrelated to id_times_dict): "
            f"{sorted(inconsistent_recs)[:10]}"
        )

    logger.info(f"Loading id_times_dict: {paths['peakIDs_Times_All']}")
    with open(paths['peakIDs_Times_All'], 'rb') as f:
        id_times_dict = pickle.load(f)

    missing_recs = set(rec_to_label.keys()) - set(id_times_dict.keys())
    if missing_recs:
        logger.warning(
            f"{len(missing_recs)} recordings in the existing pickle are NOT present in "
            f"id_times_dict at all: {sorted(missing_recs)[:10]}"
        )

    reconstructed = reconstruct_from_id_times(id_times_dict, rec_to_label)

    only_in_existing = all_existing - reconstructed
    only_in_reconstructed = reconstructed - all_existing

    logger.info(f"Reconstructed from id_times_dict: {len(reconstructed)} total samples")
    logger.info(f"In existing pickle but NOT reconstructable from id_times_dict: {len(only_in_existing)}")
    logger.info(f"In id_times_dict but NOT in existing pickle: {len(only_in_reconstructed)}")

    if only_in_existing:
        logger.warning(f"Sample of existing-only mismatches: {sorted(only_in_existing)[:10]}")
    if only_in_reconstructed:
        logger.warning(f"Sample of id_times_dict-only mismatches: {sorted(only_in_reconstructed)[:10]}")

    if not only_in_existing and not only_in_reconstructed and not missing_recs:
        logger.info(
            "MATCH: id_times_dict fully reproduces the existing pickle's "
            "(recording, frame, label) universe. Safe to proceed with the logTs removal."
        )
        return 0

    logger.error(
        "MISMATCH: id_times_dict does NOT fully reproduce the existing pickle. "
        "Do not remove the logTs directory/branch until this is understood."
    )
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
