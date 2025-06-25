import yaml
from pathlib import Path
import torch


def load_config(config_path="/rhomes/aabdel/DrosoEmbedding/src/utils/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # === Expand dynamic paths ===
    root = Path(config["paths"]["root"])
    split = config["data"]["split_strategy"]
    pickle_id = config["data"]["pickle_id"]
    neuropil = config["data"]["neuropil"]

    if neuropil == "WT":
        logT_name = "logTs"
        allT_name = "allTs"
    else:
        logT_name = f"logTs_KO_{neuropil}"
        allT_name = f"allTs_KO_{neuropil}" 

    config["paths"]["recodings_df"] = f"{config["paths"]["data_root"]}/PaulRecordings_df.xlsx"
    config["paths"]["imgs4DL"] = f"{config["paths"]["data_root"]}/imgs4DL"

    config["paths"]["pickle_path"] = str(root / config["paths"]["pickle_base"] / split / f"meanZ_{logT_name}_{pickle_id}.pickle")
    config["paths"]["allTs_path"] = f'{config["paths"]["allTs_base"]}/meanZ_{allT_name}'
    config["paths"]["results_root"] = str(root / "results" / f"{config['data']['task']}_{config['run_id']}")
    config["paths"]["models"] = f"{config['paths']['results_root']}/models/"
    config["paths"]["visualizations"] = f"{config['paths']['results_root']}/visualizations/"
    config["paths"]["evaluation"] = f"{config['paths']['results_root']}/evaluation/"



    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config["model"]["parameters"]["seq_len"] = int(config['data']['sequence']["length"])
    config["model"]["parameters"]["seq_steps"] = int(config['data']['sequence']["steps"])
    config["model"]["parameters"]["nr_classes"] = int(len(config["data"]["classes"]))


    return config
