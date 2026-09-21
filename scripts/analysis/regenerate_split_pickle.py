"""
One-time, local regeneration of the full train/val/test pickle
(paths['pickle_path']) that run_evaluation.py / training.py expect, built
entirely from what's published on G-Node:
  - IDs_logTs.pickle       (peak times)
  - Recordings_df.xlsx     (unused directly here, but required by
                             collect_image_paths()'s callers historically --
                             kept for parity)
  - data/splits/val_test_assignment_{pickle_id}.pickle  (the minimal,
    portable val/test split we extracted and verified earlier)
  - the extracted frame directory itself (paths['allTs_path'])

We deliberately did NOT publish the old full-format pickle (server-specific
absolute paths, several MB, redundant with the above). This script is the
other half of that decision: regenerate it locally, on demand, from the
portable artifacts, using split_train_val_test_from_assignment() (already
written, previously unwired). Train-set membership and all labels are
100% deterministic and recomputed fresh; only val/test membership is read
from the assignment file, because that's the one genuinely irreproducible
piece (depends on os.walk() order at original split time).

Usage (from repo root, needs the extracted frames + published pickles):
    TASK=State_Modality_Valence_16 python -m scripts.analysis.regenerate_split_pickle
    TASK=MetabolicState_2 python -m scripts.analysis.regenerate_split_pickle
    TASK=State_Modality_6 python -m scripts.analysis.regenerate_split_pickle
"""

import glob
import os
import pickle

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.preprocessing.XY_WithinAnimalSplitting import (
    collect_image_paths, filterRecordings_and_returnLabels,
    split_train_val_test_from_assignment,
)


def main():
    config = load_config()
    logger = setup_logger(task_name='regenerate_split_pickle', log_dir='logs/regenerate_split_pickle')
    paths = config['paths']
    args = config['data']['preprocessing']

    logger.info(f"Task: {config['data']['task']}")
    logger.info(f"Effective pickle_id: {config['data']['pickle_id']}")
    logger.info(f"Target pickle_path: {paths['pickle_path']}")

    assignment_path = os.environ.get(
        'VAL_TEST_ASSIGNMENT_IN',
        f"{paths['data_root']}/splits/val_test_assignment_{config['data']['pickle_id']}.pickle",
    )
    matches = glob.glob(assignment_path)
    if not matches:
        raise FileNotFoundError(
            f"No val/test assignment found at {assignment_path}. "
            "Set VAL_TEST_ASSIGNMENT_IN, or check data/splits/ for the right pickle_id."
        )
    logger.info(f"Loading val/test assignment: {matches[0]}")
    with open(matches[0], 'rb') as f:
        val_test_assignment = pickle.load(f)

    logger.info(f"Loading id_times_dict: {paths['peakIDs_Times_All']}")
    with open(paths['peakIDs_Times_All'], 'rb') as f:
        id_times_dict = pickle.load(f)

    processed_data_path = f'{paths["imgs4DL"]}/{args["method_ch"]}_allTs' if paths.get("imgs4DL") else None
    if not processed_data_path or not os.path.isdir(processed_data_path):
        # Published layout: frames live directly under allTs_path (extracted
        # from <name>.tars/), not under a paths['imgs4DL'] cluster tree.
        processed_data_path = paths['allTs_path']
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

    X_train, X_test, X_val, Y_train, Y_val, Y_test, missing = split_train_val_test_from_assignment(
        filtered_dict, labels, args['train_startingFrame'], val_test_assignment,
    )
    logger.info(f"Regenerated -- train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}, missing: {len(missing)}")
    if missing:
        logger.warning(
            f"{len(missing)} val-eligible frames have no entry in the assignment file "
            f"(sample: {missing[:10]}) -- this pickle is INCOMPLETE, do not use it for evaluation "
            "until this is understood."
        )
        return 1

    os.makedirs(os.path.dirname(paths['pickle_path']), exist_ok=True)
    with open(paths['pickle_path'], 'wb') as f:
        pickle.dump((X_train, X_val, X_test, Y_train, Y_val, Y_test), f)
    logger.info(f"Wrote regenerated split pickle to {paths['pickle_path']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
