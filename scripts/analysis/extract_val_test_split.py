"""
Extract the minimal, portable val/test split assignment from the existing
published train/val/test pickle (paths['pickle_path']), and verify it's
sufficient to exactly reconstruct the original split using only
IDs_logTs.pickle + Recordings_df.xlsx + the deterministic parts of
XY_WithinAnimalSplitting.py.

Read-only on the existing pickle -- never touched/regenerated. Writes a
new, much smaller {recording_id: {frame_number: 'val'|'test'}} mapping
elsewhere, but only after verifying it round-trips exactly.

Why this exists: the existing pickle's train-set membership and every
label are fully deterministic (a frame-index threshold + regex label
matching on recording IDs) -- reproducible from IDs_logTs.pickle +
Recordings_df.xlsx alone, on any machine, with no extra pickle. Only the
val/test 50/50 split is genuinely irreproducible: it comes from
sklearn.train_test_split(random_state=42), whose result depends on input
list order, which depends on collect_image_paths()'s os.walk()
directory-listing order -- filesystem-dependent, not portable across
machines. This script captures just that one irreducible fact, instead of
shipping a several-MB pickle full of server-specific absolute paths
(/localscratch/aabdel/...) for information that's otherwise redundant.

Usage (from repo root, needs real data + the existing pickle -- i.e. on
the cluster, not against the published drive):
    python -m scripts.analysis.extract_val_test_split
    VAL_TEST_ASSIGNMENT_OUT=scratch.pickle python -m scripts.analysis.extract_val_test_split
"""

import os
import pickle

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.preprocessing.XY_WithinAnimalSplitting import (
    collect_image_paths, filterRecordings_and_returnLabels, parse_rec_frame,
)


def build_assignment(paths, split_name):
    assignment = {}
    for p in paths:
        rec_id, frame_num = parse_rec_frame(p)
        assignment.setdefault(rec_id, {})[frame_num] = split_name
    return assignment


def main():
    config = load_config()
    logger = setup_logger(task_name='extract_val_test_split', log_dir='logs/extract_val_test_split')
    paths = config['paths']
    args = config['data']['preprocessing']

    logger.info(f"Task: {config['data']['task']}")
    logger.info(f"Effective pickle_id: {config['data']['pickle_id']}")
    logger.info(f"Effective split_by: {args['split_by']}")
    logger.info(f"Effective include_StimType={args['include_StimType']} "
                f"include_Valence={args['include_Valence']} "
                f"include_MetaboliteState={args['include_MetaboliteState']} "
                f"exclude_controls={args['exclude_controls']}")
    logger.info(f"Loading existing split pickle: {paths['pickle_path']}")
    with open(paths['pickle_path'], 'rb') as f:
        X_train_orig, X_val_orig, X_test_orig, Y_train_orig, Y_val_orig, Y_test_orig = pickle.load(f)
    logger.info(f"Existing pickle sizes -- train: {len(X_train_orig)}, val: {len(X_val_orig)}, test: {len(X_test_orig)}")

    val_test_assignment = build_assignment(X_val_orig, 'val')
    for rec_id, frames in build_assignment(X_test_orig, 'test').items():
        val_test_assignment.setdefault(rec_id, {}).update(frames)
    n_assigned = sum(len(v) for v in val_test_assignment.values())
    logger.info(f"Extracted assignment: {n_assigned} val/test frames across {len(val_test_assignment)} recordings")

    # --- Verify: recompute train + labels fresh, reconstruct the split from
    #     the extracted assignment, and compare against the original pickle ---
    logger.info(f"Loading id_times_dict: {paths['peakIDs_Times_All']}")
    with open(paths['peakIDs_Times_All'], 'rb') as f:
        id_times_dict = pickle.load(f)

    processed_data_path = f'{paths["imgs4DL"]}/{args["method_ch"]}_allTs'
    logger.info(f"Calling collect_image_paths() against: {processed_data_path}")
    rec_paths_dict = collect_image_paths(
        processed_data_path, id_times_dict, args['times'], args['exclude_controls'], logger,
    )
    filtered_dict, labels = filterRecordings_and_returnLabels(
        rec_paths_dict, args['split_by'],
        include_StimType=args['include_StimType'],
        include_Valence=args['include_Valence'],
        include_MetaboliteState=args['include_MetaboliteState'],
    )

    train_startingFrame = args['train_startingFrame']
    X_train_new, Y_train_new = [], []
    X_val_new, Y_val_new = [], []
    X_test_new, Y_test_new = [], []
    missing = []

    for recID, recPaths, label in zip(filtered_dict.keys(), filtered_dict.values(), labels):
        for path in recPaths:
            _, frameNr = parse_rec_frame(path)
            if frameNr > train_startingFrame:
                X_train_new.append(path)
                Y_train_new.append(label)
            else:
                split = val_test_assignment.get(recID, {}).get(frameNr)
                if split == 'val':
                    X_val_new.append(path); Y_val_new.append(label)
                elif split == 'test':
                    X_test_new.append(path); Y_test_new.append(label)
                else:
                    missing.append((recID, frameNr))

    def as_set(X, Y):
        return {(parse_rec_frame(p), y) for p, y in zip(X, Y)}

    train_orig_set = as_set(X_train_orig, Y_train_orig)
    val_orig_set   = as_set(X_val_orig, Y_val_orig)
    test_orig_set  = as_set(X_test_orig, Y_test_orig)
    train_new_set  = as_set(X_train_new, Y_train_new)
    val_new_set    = as_set(X_val_new, Y_val_new)
    test_new_set   = as_set(X_test_new, Y_test_new)

    train_match = train_new_set == train_orig_set
    val_match   = val_new_set == val_orig_set
    test_match  = test_new_set == test_orig_set

    logger.info(f"Train set match: {train_match} (recomputed {len(X_train_new)} vs existing {len(X_train_orig)})")
    logger.info(f"Val set match:   {val_match} (reconstructed {len(X_val_new)} vs existing {len(X_val_orig)})")
    logger.info(f"Test set match:  {test_match} (reconstructed {len(X_test_new)} vs existing {len(X_test_orig)})")
    logger.info(f"Val-eligible frames with no assignment entry: {len(missing)}")
    if not train_match:
        logger.warning(f"Train mismatch sample: only_in_existing={sorted(train_orig_set - train_new_set)[:5]} "
                        f"only_in_recomputed={sorted(train_new_set - train_orig_set)[:5]}")
    if missing:
        logger.warning(f"Missing sample: {missing[:10]}")

    if not (train_match and val_match and test_match) or missing:
        logger.error("MISMATCH -- do not publish the extracted assignment until this is understood.")
        return 1

    out_path = os.environ.get(
        'VAL_TEST_ASSIGNMENT_OUT',
        f"results/preprocessing/val_test_assignment_{config['data']['pickle_id']}.pickle",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(val_test_assignment, f)
    logger.info(f"MATCH: wrote minimal, portable val/test assignment ({n_assigned} frames, "
                f"no server-specific paths) to {out_path}")
    logger.info("This is safe to publish in place of the full train/val/test pickle -- see "
                "split_train_val_test_from_assignment() in XY_WithinAnimalSplitting.py to "
                "reconstruct the full split from it.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
