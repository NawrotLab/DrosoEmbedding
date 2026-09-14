import yaml
from pathlib import Path
import torch
import os
import random
import numpy as np
from dotenv import load_dotenv

def load_config(config_path=None):
    """
    Load configuration from YAML file with environment variable overrides.

    Environment variables that can be used to override config values:
    - RUN_ID: Overrides the run_id
    - TASK: Overrides data.task
    - BATCH_SIZE: Overrides training.batch_size
    - LEARNING_RATE: Overrides training.learning_rate
    - EPOCHS: Overrides training.epochs
    """
    load_dotenv()
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    # Load base config from YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Apply environment variable overrides
    apply_env_overrides(config)
    
    # Set up derived parameters
    setup_derived_parameters(config)
    
    return config

def apply_env_overrides(config):
    """Apply environment variable overrides to the config."""
    env_mappings = {
        'RUN_ID': ['run_id'],
        'TASK': ['data', 'task'],
        'BATCH_SIZE': ['training', 'batch_size'],
        'LEARNING_RATE': ['training', 'learning_rate'],
        'EPOCHS': ['training', 'epochs'],
        'CNN_DIM': ['model', 'parameters', 'cnn_embed_dim'],
        'TRF_DIM': ['model', 'parameters', 'transformer_embed_dim'],
        'METHOD_CH': ['data', 'preprocessing', 'method_ch'],
        'TIMES': ['data', 'preprocessing', 'times'],
        'NEUROPIL': ['data', 'preprocessing', 'neuropil'],
        'DROSO_ROOT': ['paths', 'root'],
        'DROSO_DATA_ROOT': ['paths', 'data_root'],
        'DROSO_ALLT_BASE': ['paths', 'allTs_base'],
        'DROSO_PEAK_IDS': ['paths', 'peakIDs_Times_All'],
        'DROSO_PICKLE_BASE': ['paths', 'pickle_base'],
        'DROSO_LOCAL_SCRATCH': ['paths', 'local_scratch_dir'],
        'ISOLATE_NEUROPIL': ['data', 'preprocessing', 'isolate_neuropil'],
        'SHUFFLE_LABELS_CONSISTENTLY': ['training', 'shuffle_labels_consistantly'],
        'SHUFFLE_LABELS_NAIVE': ['training', 'shuffle_labels_naive'],
        'SPLIT_BY': ['data', 'preprocessing', 'split_by'],
        'INCLUDE_STIM_TYPE': ['data', 'preprocessing', 'include_StimType'],
        'INCLUDE_VALENCE': ['data', 'preprocessing', 'include_Valence'],
        'INCLUDE_METABOLITE_STATE': ['data', 'preprocessing', 'include_MetaboliteState'],
        }
    
    for env_var, config_path in env_mappings.items():
        if env_var in os.environ:
            current = config
            for key in config_path[:-1]:
                current = current[key]
            
            # Convert the value to the appropriate type
            original_value = current[config_path[-1]]
            if isinstance(original_value, bool):
                current[config_path[-1]] = os.environ[env_var].lower() in ('true', '1', 't')
            elif isinstance(original_value, int):
                current[config_path[-1]] = int(os.environ[env_var])
            elif isinstance(original_value, float):
                current[config_path[-1]] = float(os.environ[env_var])
            elif isinstance(original_value, list):
                current[config_path[-1]] = os.environ[env_var].split(',')
            else:
                current[config_path[-1]] = os.environ[env_var]

def setup_derived_parameters(config):
    """Set up derived parameters based on the config."""
    # Set device
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # Set up task-specific parameters
    task_name = config["data"]['task']
    if task_name == 'MetabolicState_2':
        config['data']['pickle_id'] = 'S_F'
        config['data']['classes'] = ["Starved", "Fed"]
        config['training']['label_smoothing'] = 0.05
    elif task_name == 'State_Modality_6':
        config['data']['pickle_id'] = 'SO_FO_ST_FT_SM_FM'
        config['data']['classes'] = [
            "Odor (S)", "Odor (F)", 
            "Taste (S)", "Taste (F)", 
            "Odor + Taste (S)", "Odor + Taste (F)"
        ]
        config['training']['label_smoothing'] = 0.1
    elif task_name == 'State_Modality_Valence_16':
        config['data']['pickle_id'] = 'SOP_SON_FOP_FON_STP_STN_FTP_FTN_SMMP_SMMN_SMCP_SMCN_FMMP_FMMN_FMCP_FMCN'
        config['data']['classes'] = [
            # 0 - 3
            "O$^{+}$ (S)", "O$^{-}$ (S)", "O$^{+}$ (F)", "O$^{-}$ (F)",
            # 4 - 7
            "T$^{+}$ (S)", "T$^{-}$ (S)", "T$^{+}$ (F)", "T$^{-}$ (F)",
            # 8 - 11
            "O$^{+}$+T$^{+}$ (S)", "O$^{-}$+T$^{-}$ (S)", "O$^{-}$+T$^{+}$ (S)", "O$^{+}$+T$^{-}$ (S)",
            # 12 - 15
            "O$^{+}$+T$^{+}$ (F)", "O$^{-}$+T$^{-}$ (F)", "O$^{-}$+T$^{+}$ (F)", "O$^{+}$+T$^{-}$ (F)"
        ]
        config['training']['label_smoothing'] = 0.1
    # Set up neuropil-specific parameters
    neuropil = config["data"]["preprocessing"]["neuropil"]
    isolate  = config["data"]["preprocessing"]["isolate_neuropil"]
    use_aligned_template = config["data"]["preprocessing"].get("use_aligned_neuropil_template", False)
    if isolate:
        if use_aligned_template:
            logT_name = f"logTs_{neuropil}-template"
            allT_name = f"allTs_{neuropil}-template"
        else:
            logT_name = f"logTs_{neuropil}"
            allT_name = f"allTs_{neuropil}"
    else:
        logT_name = "logTs"
        allT_name = "allTs"


    config["model"]["parameters"]["seq_len"] = int(config['data']['sequence']["seq_len"])
    config["model"]["parameters"]["seq_steps"] = int(config['data']['sequence']["seq_steps"])
    config["model"]["parameters"]["nr_classes"] = int(len(config["data"]["classes"]))


    # === Expand dynamic paths ===
    root = Path(config["paths"]["root"])
    config["paths"]["checkpoints_dir"] = str(root / "results" / "_chkpt_finals")
    config["paths"]["src_imgs_dir"]    = str(root / "src" / "src_imgs")
    config["paths"]["output_dir"]      = str(root / "results" / "CombiPlots")
    config["paths"]["recodings_df"] = f"{config['paths']['data_root']}/Recordings_df.xlsx"
    config["paths"]["imgs4DL"] = f"{config['paths']['data_root']}/imgs4DL"
    # Raw recordings directory. Historically a literal "Paul_LFM_Data"
    # subfolder under data_root on the cluster; the published G-Node clone
    # names this "raw_recordings" instead -- DROSO_RAW_RECORDINGS overrides
    # directly for that case, default preserves existing cluster behavior.
    config["paths"]["raw_recordings"] = os.environ.get(
        "DROSO_RAW_RECORDINGS", f"{config['paths']['data_root']}/Paul_LFM_Data")
    config["paths"]["pickle_path"] = str(root / config["paths"]["pickle_base"] / f"meanZ_logTs_{config['data']['pickle_id']}.pickle")
    config["paths"]["pickle_path_shuffled"] = str(root / config["paths"]["pickle_base"] / f"SHUFFLED_meanZ_{logT_name}_{config['data']['pickle_id']}.pickle")
    config["paths"]["allTs_path"] = f'{config["paths"]["allTs_base"]}/meanZ_{allT_name}'
    config["paths"]["results_root"] = str(root / "results" / "model_runs" / f"{config['data']['task']}_{config['run_id']}")
    config["paths"]["models"] = f"{config['paths']['results_root']}/models/"
    config["paths"]["visualizations"] = f"{config['paths']['results_root']}/visualizations/"
    config["paths"]["evaluation"] = f"{config['paths']['results_root']}/evaluation/"

    # Published, flat, task-level layout -- where checkpoints/evaluation
    # results actually live (models/<task>/, evaluation/<task>/), as opposed
    # to the per-run results/{task}_{run_id}/ scratch tree above. Their base
    # defaults to DROSO_PUBLISH_ROOT (falling back to DROSO_ROOT if unset)
    # rather than always following DROSO_ROOT -- so DROSO_ROOT can point at
    # a normal code checkout (results/, logs, local pickle cache all stay
    # there, as scratch working state) while DROSO_PUBLISH_ROOT points at a
    # published data clone (e.g. the G-Node clone), and only the curated
    # checkpoints/evaluation results land there directly. Each is also
    # independently overridable on top of that via DROSO_MODELS_DIR/
    # DROSO_EVAL_DIR, for layouts that don't fit either pattern.
    publish_root = os.environ.get("DROSO_PUBLISH_ROOT", str(root))
    config["paths"]["models_dir"] = os.environ.get(
        "DROSO_MODELS_DIR", f"{publish_root}/models/{config['data']['task']}/")
    config["paths"]["eval_dir"] = os.environ.get(
        "DROSO_EVAL_DIR", f"{publish_root}/evaluation/{config['data']['task']}/")
    # Parent of every task's eval_dir -- what load_all_results() walks to
    # find all 3 tasks at once (evaluation/<task>/ per task), replacing the
    # old paths['checkpoints_dir'] (_chkpt_finals) as that function's input.
    config["paths"]["eval_base_dir"] = os.environ.get(
        "DROSO_EVAL_BASE_DIR", f"{publish_root}/evaluation/")
    config["paths"]["mlflow_dir"] = os.environ.get(
        "DROSO_MLFLOW_ROOT", f"{root}/mlflow")

    rng = np.random.default_rng(777)
    seeds = rng.integers(0, 2**32, size=100)
    config["seeds"] = seeds

# For backward compatibility
if __name__ == "__main__":
    config = load_config()
    print("Configuration loaded successfully.")
