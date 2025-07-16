import yaml
from pathlib import Path
import torch


def load_config(config_path="/rhomes/aabdel/DrosoEmbedding/src/utils/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    #=== Define some vars ===
    split = config['data']['split_strategy']




    # === dynamically set up more parameters ===
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task_name = config["data"]['task']
    if task_name == 'MetabolicState_2':
        config['data']['pickle_id'] = 'S_F'
        config['data']['classes'] = ["Starved", "Fed"]
    elif task_name == 'State_Modality_6':
        config['data']['pickle_id'] = 'SO_FO_ST_FT_SM_FM'
        config['data']['classes'] = ["Odor (S)","Odor (F)", 
                                     "Taste (S)", "Taste (F)", 
                                     "Odor + Taste (S)", "Odor + Taste (F)"]
    elif task_name == 'State_Modality_Valence_16':
        config['data']['pickle_id'] = 'SOP_SON_FOP_FON_STP_STN_FTP_FTN_SMMP_SMMN_SMCP_SMCN_FMMP_FMMN_FMCP_FMCN'
        config['data']['classes'] = [
            #  0 -  3
            "O$^{+}$ (S)", "O$^{-}$ (S)", "O$^{+}$ (F)", "O$^{-}$ (F)",
            #  4 -  7
            "T$^{+}$ (S)", "T$^{-}$ (S)", "T$^{+}$ (F)", "T$^{-}$ (F)",
            #  8 - 11
            "O$^{+}$+T$^{+}$ (S)", "O$^{-}$+T$^{-}$ (S)", "O$^{-}$+T$^{+}$ (S)", "O$^{+}$+T$^{-}$ (S)",
            # 12 - 16
            "O$^{+}$+T$^{+}$ (F)", "O$^{-}$+T$^{-}$ (F)", "O$^{-}$+T$^{+}$ (F)", "O$^{+}$+T$^{-}$ (F)"]
  
    neuropil = config["data"]["preprocessing"]["neuropil"]
    if neuropil == "WT":
        logT_name = "logTs"
        allT_name = "allTs"
    else:
        logT_name = f"logTs_KO_{neuropil}"
        allT_name = f"allTs_KO_{neuropil}"
     
    config["model"]["parameters"]["seq_len"] = int(config['data']['sequence']["seq_len"])
    config["model"]["parameters"]["seq_steps"] = int(config['data']['sequence']["seq_steps"])
    config["model"]["parameters"]["nr_classes"] = int(len(config["data"]["classes"]))


    # === Expand dynamic paths ===
    root = Path(config["paths"]["root"])
    config["paths"]["recodings_df"] = f"{config["paths"]["data_root"]}/PaulRecordings_df.xlsx"
    config["paths"]["imgs4DL"] = f"{config["paths"]["data_root"]}/imgs4DL"
    config["paths"]["pickle_path"] = str(root / config["paths"]["pickle_base"] / split / f"meanZ_{logT_name}_{config['data']['pickle_id']}.pickle")
    config["paths"]["pickle_path_shuffled"] = str(root / config["paths"]["pickle_base"] / split / f"SHUFFLED_meanZ_{logT_name}_{config['data']['pickle_id']}.pickle")
    config["paths"]["allTs_path"] = f'{config["paths"]["allTs_base"]}/meanZ_{allT_name}'
    config["paths"]["results_root"] = str(root / "results" / f"{config['data']['task']}_{config['run_id']}")
    config["paths"]["models"] = f"{config['paths']['results_root']}/models/"
    config["paths"]["visualizations"] = f"{config['paths']['results_root']}/visualizations/"
    config["paths"]["evaluation"] = f"{config['paths']['results_root']}/evaluation/"


    return config
