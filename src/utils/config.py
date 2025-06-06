import os
from datetime import datetime
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, GradCAMElementWise


# classes = ['SOP', 'SON', 'FOP', 'FON', 'STP', 'STN', 'FTP', 'FTN', 'SMMP', 'SMMN', 'FMMP', 'FMMN']
classes = ['Starved', 'Fed']
# classes = ['Odor', 'Taste', 'Odor + Taste']
# classes = ['Positive', 'Negative']
# classes = ['Appetitive', 'Aversive']

# classes = ["Positive Stimulus", "Negative Stimulus"]     # SON
#
# classes = [
#     "Starved-Odor (+)",     # SOP
#     "Starved-Odor (-)",     # SON
#     "Fed-Odor (+)",         # FOP
#     "Fed-Odor (-)",         # FON
#     "Starved-Taste (+)",    # STP
#     "Starved-Taste (-)",    # STN
#     "Fed-Taste (+)",        # FTP
#     "Fed-Taste (-)",        # FTN
#     "Starved-Multi (+)",    # SMMP
#     "Starved-Multi (-)",    # SMMN
#     "Fed-Multi (+)",        # FMMP
#     "Fed-Multi (-)"         # FMMN
# ]
#
# classes = [
#     # Odor-Only Conditions
#     "Odor$^{+}$ (S)",  # SOP
#     "Odor$^{-}$ (S)",  # SON
#     "Odor$^{+}$ (F)",  # FOP
#     "Odor$^{-}$ (F)",  # FON
#
#     # Taste-Only Conditions
#     "Taste$^{+}$ (S)",  # STP
#     "Taste$^{-}$ (S)",  # STN
#     "Taste$^{+}$ (F)",  # FTP
#     "Taste$^{-}$ (F)",  # FTN
#
#     # Multi (Bimodal: Odor + Taste)
#     "Odor$^{+}$+Taste$^{+}$ (S)",  # SMMP
#     "Odor$^{-}$+Taste$^{-}$ (S)",  # SMMN
#     "Odor$^{+}$+Taste$^{+}$ (F)",
#     "Odor$^{-}$+Taste$^{-}$ (F)",
# ]



# classes = [
#     # Multi (Bimodal: Odor + Taste)
#     "O$^{(+)}$+T$^{(+)}$ (S)",
#     "O$^{(+)}$+T$^{(+)}$ (F)",
#     "O$^{(-)}$+T$^{(-)}$ (S)",
#     "O$^{(-)}$+T$^{(-)}$ (F)",
#
#     "O$^{(-)}$+T$^{(+)}$ (S)",
#     "O$^{(-)}$+T$^{(+)}$ (F)",
#     "O$^{(+)}$+T$^{(-)}$ (S)",
#     "O$^{(+)}$+T$^{(-)}$ (F)"
# ]


# classes = [
#     # Multi (Bimodal: Odor + Taste)
#     "Odor$^{+}$+\nTaste$^{+}$ (S)",
#     "Odor$^{+}$+\nTaste$^{+}$ (F)",
#     "Odor$^{-}$+\nTaste$^{-}$ (S)",
#     "Odor$^{-}$+\nTaste$^{-}$ (F)",

#     "Odor$^{-}$+\nTaste$^{+}$ (S)",
#     "Odor$^{-}$+\nTaste$^{+}$ (F)",
#     "Odor$^{+}$+\nTaste$^{-}$ (S)",
#     "Odor$^{+}$+\nTaste$^{-}$ (F)"
# ]
#
# classes = [
#     # Multi (Bimodal: Odor + Taste)
#     "(S)",
#     "(F)",
#     "(S)",
#     "(F)",
#
#     "(S)",
#     "(F)",
#     "(S)",
#     "(F)"
# ]

# classes = [
#     "(S)", "(S)", "(F)",  "(F)", # Odor-Only Conditions
#     "(S)", "(S)", "(F)", "(F)",  # Taste-Only Conditions
#     "(S)", "(S)", "(F)", "(F)", # Multi (Bimodal: Odor + Taste)
# ]

training_config = {
    "model_name": "CNN_Transformer",

    "run_name" : "2MCLASSES_10eps_newLogger",
    "pickle_ID_name": "S_F",

    "ROOT_PATH": '/rhomes/aabdel/MSc_DL4DrosoWBCI',
    "class_names": classes,
    "split_strategy": 'SplitWithinAnimal',
    "model_params": {'nr_channels': 1,
                     'embed_dim': 16,
                     'num_heads': 2, #8, #4,
                     'num_layers': 1, #6, #2,
                     'nr_classes': len(classes),
                     'seq_len': 5,
                     'seq_steps': 10, #5,
                     'num_epochs': 10,
                     'lr': 0.0005, #0.0005
                     "weight_decay":1e-5, # was 1e-5
                     "criterion" : nn.CrossEntropyLoss()},
    "batch_size": 256,  #256,
    "plot_batch_size" : 16,
    "CAM_method" : GradCAMPlusPlus, #ScoreCAM,
    "cam_targetLayer" : 0,
    "Neuropil": "WT",  #"WT"
}




if training_config["Neuropil"] == "WT":
    logT_name = "logTs"
    allT_name = "allTs"
else:
    logT_name = f"logTs_KO_{training_config["Neuropil"]}"
    allT_name = f"allTs_KO_{training_config["Neuropil"]}"
training_config["pickleID_logTs"] = f'meanZ_{logT_name}_{training_config["pickle_ID_name"]}'
training_config["data_PicklePath"] = f'{training_config["ROOT_PATH"]}/pickles/TrainValTest_LSImagePaths_Labels/{training_config["split_strategy"]}/{training_config["pickleID_logTs"]}.pickle'
training_config["data_path_allTs"] = f'/localscratch/aabdel/imgs4DL/meanZ_{allT_name}'
training_config["evaluation_path"] = f'{training_config["ROOT_PATH"]}/results/{training_config["model_name"]}_{training_config["run_name"]}/evaluation/'
training_config["visualizations_path"] = f'{training_config["ROOT_PATH"]}/results/{training_config["model_name"]}_{training_config["run_name"]}/visualizations/'
training_config["model_root_path"] = f'{training_config["ROOT_PATH"]}/results/{training_config["model_name"]}_{training_config["run_name"]}/models/'


class_names = [
            "Starved_Odor_Positive", "Starved_Odor_Negative",
            "Fed_Odor_Positive", "Fed_Odor_Negative",
            "Starved_Taste_Positive", "Starved_Taste_Negative",
            "Fed_Taste_Positive", "Fed_Taste_Negative",
            "Starved_Mix_Positive", "Starved_Mix_Negative",
            "Fed_Mix_Positive", "Fed_Mix_Negative"
        ]
class_names = ["Positive_Stimulus", "Negative_Stimulus"]
nr_classes = len(class_names)




config = {
    "run_name": "CNN_Transformer",
    "split_strategy": "SplitWithinAnimal",
    "ROOT_PATH": '/rhomes/aabdel/MSc_DL4DrosoWBCI/pickles/TrainValTest_ImagePaths_Labels/SplitWithinAnimal',
    "allTs_path" :'/projects/lab-data/Collaboration/Gruenwald_Kadow/imgs4DL/meanZ_allTs',
    # "dataID_logTs": "meanZ_logTs_SOP_SON_FOP_FON_STP_STN_FTP_FTN_SMMP_SMMN_FMMP_FMMN",
    # "dataID_allTs": "meanZ_allTs_SOP_SON_FOP_FON_STP_STN_FTP_FTN_SMMP_SMMN_FMMP_FMMN",
    "dataID_logTs": "meanZ_logTs_P_N",
    "dataID_allTs": "meanZ_allTs_P_N",
    "class_names": class_names,
    "nr_classes": nr_classes,
    "plot_TSNE": True,
    # "AutoCast": True,
    # "SupConLoss": False,
    # "ceTemperature":0.01,
    "seq_length": 5,
    "seq_steps": 5,
    "num_epochs": 5000,
    "batch_size": 256, #256,
    "lr": 0.0005,
    "dropout": 0.3, # 0.3,
    "num_layers": 1,
    "embed_dim": 16,
    "num_heads": 2,
    "nr_channels": 1,
    "criterion": "cross_entropy",
    "num_workers":8,
    "plot_batch_size": 128,
    "cam_targetLayer": 3,
    "CAM_method": "ScoreCAM",  # can be GradCAM, etc.
    "advanced_learning_dynamics": False, #was true
    "cache_images": True,
    "date": "2Classes_Valence"#"12Classes_00DO" #"noceLoss_noAutoCast" #datetime.now().strftime("%Y%m%d") # "20250107"
}


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config["device"] = device
training_config["device"] = device

# Preprocessing Parameters
preprocessingID = 'meanZ_allTs'
preprocessing_variables = {
    'outputID': f'{preprocessingID}',
    'ROOT_PATH': '/projects/lab-data/Collaboration/Gruenwald_Kadow',
    'path_data': '/projects/lab-data/Collaboration/Gruenwald_Kadow/Paul_LFM_Data',
    'path_output': f'/projects/lab-data/Collaboration/Gruenwald_Kadow/imgs4DL/{preprocessingID}',
    # 'path_entireRecs': '/projects/lab-data/Collaboration/Gruenwald_Kadow/imgs4DL/KO_niftis/',
    'path_baseFrame': f'/projects/lab-data/Collaboration/Gruenwald_Kadow/imgs4DL/{preprocessingID}/BaseFrames/',
    'path_pngs': f'/projects/lab-data/Collaboration/Gruenwald_Kadow/imgs4DL/{preprocessingID}/pngs/',
    'log_dir' : '/rhomes/aabdel/MSc_DL4DrosoWBCI/logs/preprocessing',
    'knock_neuropil_out': True,
    'neuropil': 'INP',
    'base_points': [0, 250]
}

config["meanZ_allTs_preprocessing"] = preprocessing_variables


# preprocessing_params = {
#     'knock_neuropil_out': True,
#     'neuropil': 'MB',
#     'PREPROCESSING_ID': 'meanZ_logTs',
#     'exclude_controls': True,
#     'base_points': [0, 250]
# }

# define model parameters
model_params = {
    'nr_channels': config['nr_channels'],
    'embed_dim': config['embed_dim'],
    'num_heads': config['num_heads'],
    'num_layers': config['num_layers'],
    'nr_classes': config['nr_classes'],
    'seq_len': config['seq_length'],
    'dropout': config['dropout']
}
config["model_params"] = model_params

# define paths
config["output_root"] = os.path.join(
    "results/",
    # config["run_name"] + '_' + config["date"] + f'_{len(class_names)}Classes' + f'{config["seq_length"]}imgs{config["seq_steps"]}steps' + '_nepochs' + str(config["num_epochs"]) + '_lr' + str(config["lr"]))
    # config["run_name"] + '_' + config["date"] + '_12Classes' + '_nepochs' + str(config["num_epochs"]))

    config["run_name"] + '_' + config["date"] + '_nepochs' + str(config["num_epochs"]))

config["model_path"] = config["output_root"] + '/models/'
config["visualizations_path"] = config["output_root"] + '/visualizations/'
config["evaluation_path"] = config["output_root"] + '/evaluation/'

