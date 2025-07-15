import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
import re
import random


def collect_image_paths(root_path, exclude_controls):
    recording_paths_dict = {}
    for root, directories, files in os.walk(root_path):
        if 'BaseFrames' in directories: directories.remove('BaseFrames')
        if 'pngs' in directories: directories.remove('pngs')
        if exclude_controls:
            directories = [d for d in directories if d[-5] != 'C']
        for rec in directories:
            current_dir = os.path.join(root, rec)
            logger.info(f'Adding images from {rec}')
            recording_paths_dict[rec] = [os.path.join(current_dir, file) for file in os.listdir(current_dir)]
    return recording_paths_dict


def filterRecordings_and_returnLabels(rec_paths_dict, classes, include_StimType, include_Valence, include_MetaboliteState):
    filtered_dictionary = {}
    labels = []

    class_to_label = {pattern: i for i, pattern in enumerate(classes)}

    if include_StimType and include_Valence and include_MetaboliteState:  # eg SOP
        stimuli = np.unique([i[1:-1] for i in classes]).tolist()
        states = np.unique([i[0] for i in classes]).tolist()
        valences = np.unique([i[-1] for i in classes]).tolist()
    elif include_StimType and include_MetaboliteState and not include_Valence:  # eg SO
        stimuli = np.unique([i[1:] for i in classes]).tolist()
        states = np.unique([i[0] for i in classes]).tolist()
    elif include_StimType and include_Valence and not include_MetaboliteState:  # eg OP
        stimuli = np.unique([i[:-1] for i in classes]).tolist()
        valences = np.unique([i[-1] for i in classes]).tolist()
    elif include_StimType and not include_Valence and not include_MetaboliteState:  # eg O
        stimuli = classes
    elif include_MetaboliteState and not include_StimType and not include_Valence: # eg S, F
        states = np.unique([i for i in classes]).tolist()
    else:
        print("menmen still has to add this case")

    for recID, img_paths in rec_paths_dict.items():
        match = re.match(r'^([A-Z]+)', recID)

        if match:
            recCond = match.group(1)
            current_StimType = recCond[1:-1]
            current_Valence = recCond[-1]
            current_MetaboliteState = recCond[0]

            # for whenever i dont care about MultiMatch or MultiContra
            if include_StimType: 
                if len(current_StimType) == 2 and 'M' in stimuli:
                    current_StimType = 'M'

            if ((not include_StimType or current_StimType in stimuli) and
                    (not include_Valence or current_Valence in valences) and
                    (not include_MetaboliteState or current_MetaboliteState in states)):
                filtered_dictionary[recID] = img_paths

                # Create the class label string based on the current conditions
                if include_StimType and include_Valence and include_MetaboliteState:
                    class_pattern = f"{current_MetaboliteState}{current_StimType}{current_Valence}"
                elif include_StimType and include_MetaboliteState and not include_Valence:
                    class_pattern = f"{current_MetaboliteState}{current_StimType}"
                elif include_StimType and include_Valence and not include_MetaboliteState:
                    class_pattern = f"{current_StimType}{current_Valence}"
                elif include_StimType and not include_Valence and not include_MetaboliteState:
                    class_pattern = current_StimType
                elif not include_StimType and not include_Valence and include_MetaboliteState:
                    class_pattern = current_MetaboliteState
                else:
                    class_pattern = None  # Handle other cases if needed

                # Lookup the label for the identified class pattern and append it
                if class_pattern in class_to_label:
                    labels.append(class_to_label[class_pattern])
                else:
                    labels.append(None)  # If class_pattern isn't found, you could also raise an error

    return filtered_dictionary, labels


def split_train_val_test(rec_paths_dict, labels, train_startingFrame, val_test_proportion):
    X_train, Y_train, X_val_test, Y_val_test = [], [], [], []

    for recID, recPaths, label in zip(rec_paths_dict.keys(), rec_paths_dict.values(), labels):
        for path in recPaths:
            frameNr = int(path.split('_')[-1].split('.')[0])
            if frameNr <= train_startingFrame:
                X_val_test.append(path)
                Y_val_test.append(label)
            elif frameNr > train_startingFrame:
                X_train.append(path)
                Y_train.append(label)

    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=val_test_proportion, random_state=42)

    return X_train, X_test, X_val, Y_train, Y_val,  Y_test

def preview_random_samples(X, Y, n=5, set_name="train"):
    logger.info(f"\nRandom {n} samples from {set_name} set:")
    indices = random.sample(range(len(X)), n)
    for i in indices:
        x = X[i]
        y = Y[i]
        logger.info(f"Index: {i}")
        logger.info(f"Y: {y}")
        if isinstance(x, np.ndarray):
            logger.info(f"X shape: {x.shape}, dtype: {x.dtype}, min: {x.min()}, max: {x.max()}")
        else:
            logger.info(f"X type: {type(x)} -> {x}")
        logger.info("-" * 40)


def main(config, logger):
        
    args = config['data']['preprocessing']
    paths = config['paths']

    PREPROCESSING_ID = f"{args['method_ch']}_{args['times']}" #args['id']
    CLASSES = args['split_by']
    outID = '_'.join(CLASSES)

    exclude_controls = args['exclude_controls']
    include_StimType = args['include_StimType']
    include_MetaboliteState = args['include_MetaboliteState']
    include_Valence = args['include_Valence']
    train_startingFrame = args['train_startingFrame']
    val_test_proportion = args['val_test_proportion']

    ROOT_PATH = paths["data_root"]
    RECORDING_LIST= paths["recodings_df"]
    PROSESSED_DATA_PATH = f'{paths["imgs4DL"]}/{PREPROCESSING_ID}'

    PICKLE_OUTPATH = f'{paths["root"]}/pickles/TrainValTest_LocalScratch_Paths-Labels/{config["data"]["split_strategy"]}/{PREPROCESSING_ID}_{outID}.pickle'

    rec_paths_dict = collect_image_paths(PROSESSED_DATA_PATH, exclude_controls=exclude_controls)
    filtered_dict, labels = filterRecordings_and_returnLabels(rec_paths_dict, CLASSES, include_StimType=include_StimType, include_Valence= include_Valence, include_MetaboliteState= include_MetaboliteState)

    if config['training']['shuffle_labels_consistantly']: 
        random.shuffle(labels)
        PICKLE_OUTPATH = f'{paths["root"]}/pickles/TrainValTest_LocalScratch_Paths-Labels/{config["data"]["split_strategy"]}/SHUFFLED_{PREPROCESSING_ID}_{outID}.pickle'

    X_train, X_test, X_val, Y_train, Y_val,  Y_test = split_train_val_test(filtered_dict, labels, train_startingFrame, val_test_proportion)

    logger.info("User please check the pairings make sense! ;-)")
    logger.info(f"We are spliiting into following: {args["split_by"]}")
    preview_random_samples(X_train, Y_train, n=2, set_name="train")
    preview_random_samples(X_val, Y_val, n=2, set_name="val")
    preview_random_samples(X_test, Y_test, n=2, set_name="test")

    with open(PICKLE_OUTPATH, 'wb') as file:
        pickle.dump((X_train, X_val, X_test, Y_train, Y_val, Y_test), file)





if __name__ == '__main__':
    config = load_config()
    logger = setup_logger(task_name=config["run_id"], log_dir="logs/slurm")
    main(config, logger)
