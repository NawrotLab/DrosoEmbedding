from __future__ import annotations
from pathlib import Path
from typing import Iterable, Mapping, Any, Dict, Optional
import os, re, pickle
import mlflow
import numpy as np
import torch
from sklearn.manifold import TSNE
import yaml
from pathlib import Path
import gc
# import umap

# Shared configuration constants
BASE_RESULTS_DIR = os.path.join('results', '_chkpt_finals')


def paths2neuropilpaths(X, config):
    prep = config["data"]["preprocessing"]
    base = "meanZ_logTs"

    # sanity check
    if prep["isolate_neuropil"] and prep["remove_neuropil"]:
        raise ValueError(
            "Both isolate_neuropil and remove_neuropil are True. Choose only one."
        )

    # decide new folder name
    if prep["isolate_neuropil"]:
        new_base = f"{base}_{prep['neuropil']}"
    elif prep["remove_neuropil"]:
        new_base = f"{base}_KO_{prep['neuropil']}"
    else:
        return X  # nothing to do

    X_new = []
    for p in X:
        p = Path(p)
        parts = list(p.parts)

        try:
            idx = parts.index(base)
        except ValueError:
            raise ValueError(f"Expected '{base}' in path but got:\n{p}")

        parts[idx] = new_base
        X_new.append(str(Path(*parts)))

    return X_new


def log_params_recursive(d):
    for k, v in d.items():
        if isinstance(v, dict):
            log_params_recursive(v)
        elif isinstance(v, (int, float, str, bool)):
            mlflow.log_param(k, v)

def get_latent_space(model, data_loader, device, return_cnn_latent=False):
    model.eval()
    latent_space, latent_labels = [], []
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences = sequences.float().to(device)
            labels = labels.to(device)
            # Pass data through the model and collect latent features
            latent_features = model(sequences, return_latent=True ,return_cnn_latent=return_cnn_latent)
            latent_space.append(latent_features.cpu().numpy())
            latent_labels.append(labels.cpu().numpy())

    # Stack collected features and labels
    latent_features_np = np.concatenate(latent_space, axis=0)
    latent_labels_np = np.concatenate(latent_labels, axis=0)
    return latent_features_np, latent_labels_np

def get_predictions(model, data_loader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences = sequences.float().to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(sequences)
            predicted = torch.argmax(outputs, dim=1)
            predictions.append(predicted.cpu().numpy())
            true_labels.append(labels.cpu().numpy())
    # Stack collected predictions and labels
    predictions_np = np.concatenate(predictions, axis=0)
    true_labels_np = np.concatenate(true_labels, axis=0)
    return predictions_np, true_labels_np 

def compute_tsne(latent_features, perplexity=30, random_state=42):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    return tsne.fit_transform(latent_features)

def _color_for_group(styles: Dict[str, Any], code: str) -> str:
    key_map = {
        'S': 'starved',
        'F': 'fed',
        'O': 'odor',
        'T': 'taste',
        'O/T': 'odor_taste',
        '+': 'positive',
        '-': 'negative',
        '+/-': 'positive_negative',
    }
    style_key = key_map.get(code, None)
    if style_key is None:
        return 'k'
    d = styles.get(style_key, {})
    return d.get('arrow_color', d.get('color', 'k'))



def _get_group_map_eval(task_name: str, class_names: list[str]) -> dict[str, list[int]]:
    if task_name == 'MetabolicState_2':
        return {'S': [0], 'F': [1]}
    
    if task_name == 'State_Modality_6':
        return {'S': [0, 2, 4], 'F': [1, 3, 5], 'O': [0, 1], 'T': [2, 3], 'O/T': [4, 5]}

    if task_name == 'State_Modality_Valence_16':
        pos_inds = [i for i, n in enumerate(class_names) if '+' in n and '-' not in n]
        neg_inds = [i for i, n in enumerate(class_names) if '-' in n and '+' not in n]
        return {
            'S': [0, 1, 4, 5, 8, 9, 10, 11],
            'F': [2, 3, 6, 7, 12, 13, 14, 15],
            'O': [0, 1, 2, 3],
            'T': [4, 5, 6, 7],
            'O/T': [8, 9, 10, 11, 12, 13],
            '+': pos_inds,
            '-': neg_inds,
            '+/-': [10, 11, 14, 15],
        }
    return None

# def load_all_results(base_dir, task_names=None):
#     """
#     Scan a results root and return:
#     {
#       'MetabolicState_2': {
#         'control': <loaded pkl or None>,
#         'best':    <loaded pkl or None>,
#         'runs': {
#           'E4':  [ {'run':1,'dimension':4,'path':..., 'data':...}, ... ],
#           'E8':  [ ... ],
#           'E16': [ ... ],
#           ...
#         }
#       },
#       'State_Modality_6': { ... },
#       'State_Modality_Valence_16': { ... }
#     }

#     Args:
#       base_dir    : path to the folder containing the task subfolders
#       task_names  : optional iterable of task folder names to include; if None, include all dirs
#     """
#     import os, re, pickle
#     from pathlib import Path

#     base = Path(base_dir)

#     # one-file regex parser; supports ..._E16_10.pkl and ..._trf16_10.pkl
#     pat1 = re.compile(r'[_-]E(?P<dim>\d+)[_-](?P<run>\d+)\.pkl$', re.IGNORECASE)
#     pat2 = re.compile(r'[_-]trf(?P<dim>\d+)[_-](?P<run>\d+)\.pkl$', re.IGNORECASE)

#     def parse_dim_run(name):
#         m = pat1.search(name) or pat2.search(name)
#         if m:
#             return int(m.group("dim")), int(m.group("run"))
#         # fallback: last two integers in the stem are (dim, run)
#         stem = os.path.splitext(name)[0]
#         nums = re.findall(r'\d+', stem)
#         if len(nums) >= 2:
#             return int(nums[-2]), int(nums[-1])
#         raise ValueError(f"dim/run not found in '{name}'")

#     def load_latest_pkl(dir_path: Path):
#         if not dir_path.is_dir():
#             return None
#         pkls = [p for p in dir_path.iterdir() if p.is_file() and p.suffix == ".pkl"]
#         if not pkls:
#             return None
#         pkls.sort(key=lambda p: p.stat().st_mtime, reverse=True)  # newest first
#         with pkls[0].open("rb") as f:
#             return pickle.load(f)

#     def load_pickle(path: Path):
#         with path.open("rb") as f:
#             return pickle.load(f)

#     # choose which task folders to walk
#     tasks = [p for p in base.iterdir() if p.is_dir()]
#     if task_names:
#         names = set(task_names)
#         tasks = [p for p in tasks if p.name in names]
#     tasks.sort(key=lambda p: p.name)

#     out = {}
#     for task_dir in tasks:
#         task = task_dir.name
#         entry = {"control": None, "best": None, "runs": {}}

#         # control (support 'control' OR 'control_run')
#         for c in ("control", "control_run"):
#             cand = task_dir / c
#             v = load_latest_pkl(cand)
#             if v is not None:
#                 entry["control"] = v
#                 break

#         # best
#         entry["best"] = load_latest_pkl(task_dir / "best")

#         # runs (support 'runs' OR 'dim_runs')
#         for runs_folder in ("runs", "dim_runs"):
#             rdir = task_dir / runs_folder
#             if not rdir.is_dir():
#                 continue
#             for p in sorted(rdir.iterdir(), key=lambda x: x.name):
#                 if not (p.is_file() and p.suffix == ".pkl"):
#                     continue
#                 try:
#                     dim, run = parse_dim_run(p.name)
#                 except Exception as e:
#                     print(f"Skipping {p.name}: {e}")
#                     continue
#                 key = f"E{dim}"
#                 entry["runs"].setdefault(key, []).append(
#                     {"run": run, "dimension": dim, "path": str(p), "data": load_pickle(p)}
#                 )

#         # sort each dimension's list by run index
#         for k in entry["runs"]:
#             entry["runs"][k].sort(key=lambda d: d["run"])

#         out[task] = entry

#         if task in task_names:
#             out[task]['__class_names__'] = task_names[task]

#     return out
# -------
# def load_all_results(
#     base_dir: str | os.PathLike,
#     task_names: Optional[Iterable[str] | Mapping[str, Any]] = None,
#     fixed_trf_for_E: int = 16,   # H must equal this to populate E* groups
#     fixed_cnn_for_H: int = 16,   # E must equal this to populate H* groups
#     logger = None,
# ) -> Dict[str, Dict[str, Any]]:
#     """
#     Scan a results root and return, per task:
#       {
#         '...': {
#           'control': <pkl or None>,
#           'best'   : <pkl or None>,
#           'runs'   : {
#             'E4' : [ {run, classes, cnn_dim, trf_dim, path, data}, ... ]   # ONLY files with H==fixed_trf_for_E
#             'E8' : [ ... ],
#             ...
#             'H4' : [ {run, classes, cnn_dim, trf_dim, path, data}, ... ]   # ONLY files with E==fixed_cnn_for_H
#             'H8' : [ ... ],
#             ...
#           },
#           '__class_names__': ...  # only if task_names is a mapping
#         }
#       }

#     Only files strictly matching 'C{classes}_E{cnn}_H{trf}_{run}.pkl' are considered.
#     """
#     base = Path(base_dir)

#     # Strict pattern: C2_E4_H16_1.pkl (underscores only, optional dash before run also allowed)
#     strict = re.compile(
#         r'^C(?P<classes>\d+)_E(?P<cnn>\d+)_H(?P<trf>\d+)[_-](?P<run>\d+)\.pkl$',
#         re.IGNORECASE
#     )

#     def parse_filename(name: str):
#         m = strict.match(name)
#         if not m:
#             raise ValueError(f"Non-matching filename '{name}'")
#         return (
#             int(m.group("classes")),
#             int(m.group("cnn")),
#             int(m.group("trf")),
#             int(m.group("run")),
#         )

#     def load_latest_pkl(dir_path: Path):
#         if not dir_path.is_dir():
#             if logger:
#                 logger.debug(f"load_latest_pkl: {dir_path} is not a directory")
#             return None
#         pkls = [p for p in dir_path.iterdir() if p.is_file() and p.suffix == ".pkl"]
#         if not pkls:
#             if logger:
#                 logger.debug(f"load_latest_pkl: No .pkl files found in {dir_path}")
#             return None
#         pkls.sort(key=lambda p: p.stat().st_mtime, reverse=True)  # newest first
#         try:
#             file_size = pkls[0].stat().st_size
#             # if logger:
#             #     logger.info(f"load_latest_pkl: Loading {pkls[0].name} ({file_size / (1024*1024):.2f} MB) from {dir_path}")
#             with pkls[0].open("rb") as f:
#                 data = pickle.load(f)
#                 # if logger:
#                 #     logger.info("data keys: {data.keys()}")
#                 #     logger.info(f"load_latest_pkl: Successfully loaded {pkls[0].name}")
#                 return data
#         except Exception as e:
#             if logger:
#                 logger.warning(f"load_latest_pkl: Failed to load latest pkl in {dir_path}: {e}")
#             else:
#                 print(f"[WARN] Failed to load latest pkl in {dir_path}: {e}")
#             return None

#     def load_pickle(path: Path):
#         try:
#             file_size = path.stat().st_size
#             # if logger:
#             #     logger.info(f"load_pickle: Loading {path.name} ({file_size / (1024*1024):.2f} MB)")
#             with path.open("rb") as f:
#                 data = pickle.load(f)
#                 if logger:
#                     logger.info(f"data keys: {data.keys()}")
#                     logger.info(f"load_pickle: Successfully loaded {path.name}")
#                 return data
#         except Exception as e:
#             if logger:
#                 logger.warning(f"load_pickle: Failed to load {path.name}: {e}")
#             else:
#                 print(f"[WARN] Failed to load {path.name}: {e}")
#             return None

#     # Normalize task_names into a whitelist and optional class-name mapping
#     whitelist: Optional[set[str]] = None
#     classmap: Optional[Mapping[str, Any]] = None
#     if task_names is not None:
#         if isinstance(task_names, Mapping):
#             classmap = task_names
#             whitelist = set(task_names.keys())
#         else:
#             whitelist = set(task_names)

#     # choose which task folders to walk
#     if logger:
#         logger.debug(f"load_all_results: Scanning base directory {base}")
#     tasks = [p for p in base.iterdir() if p.is_dir()]
#     if whitelist is not None:
#         tasks = [p for p in tasks if p.name in whitelist]
#     tasks.sort(key=lambda p: p.name)
#     if logger:
#         logger.debug(f"load_all_results: Found {len(tasks)} task directories: {[t.name for t in tasks]}")

#     out: Dict[str, Dict[str, Any]] = {}
#     for task_dir in tasks:
#         task = task_dir.name
#         if logger:
#             logger.debug(f"load_all_results: Processing task '{task}'")
#         entry: Dict[str, Any] = {"control": None, "best": None, "runs": {}}

#         # control (support 'control' OR 'control_run')
#         for c in ("control", "control_run"):
#             cand = task_dir / c
#             v = load_latest_pkl(cand)
#             if v is not None:
#                 entry["control"] = v
#                 if logger:
#                     logger.debug(f"load_all_results: Found control file for task '{task}' in {cand}")
#                 break

#         # best
#         best_path = task_dir / "best"
#         entry["best"] = load_latest_pkl(best_path)
#         if entry["best"] is not None and logger:
#             logger.debug(f"load_all_results: Found best file for task '{task}'")

#         # runs (support 'runs' OR 'dim_runs'), but ONLY strict filenames C*_E*_H*_* .pkl
#         for runs_folder in ("runs", "dim_runs"):
#             rdir = task_dir / runs_folder
#             if not rdir.is_dir():
#                 continue
#             if logger:
#                 logger.info(f"load_all_results: Processing runs folder '{runs_folder}' for task '{task}'")
#             pkl_files = [p for p in rdir.iterdir() if p.is_file() and p.suffix == ".pkl"]
#             if logger:
#                 logger.info(f"load_all_results: Found {len(pkl_files)} .pkl files in {rdir}")

#             file_count = 0
#             for p in sorted(pkl_files, key=lambda x: x.name):
#                 file_count += 1
#                 try:
#                     classes, cnn_dim, trf_dim, run = parse_filename(p.name)
#                     file_size = p.stat().st_size
#                 except Exception as e:
#                     # Strictly skip anything not matching the new scheme
#                     if logger:
#                         logger.debug(f"load_all_results: Skipping {p.name}: {e}")
#                     else:
#                         print(f"[INFO] Skipping {p.name}: {e}")
#                     continue

#                 if logger:
#                     logger.info(f"load_all_results: Processing file {file_count}/{len(pkl_files)}: {p.name} ({file_size / (1024*1024):.2f} MB)")
#                 rec = {
#                     "run": run,
#                     "classes": classes,
#                     "cnn_dim": cnn_dim,   # E*
#                     "trf_dim": trf_dim,   # H*
#                     "path": str(p),
#                     "data": load_pickle(p),
#                 }
#                 if rec["data"] is None:
#                     if logger:
#                         logger.warning(f"load_all_results: Failed to load data from {p.name}, skipping")
#                     continue
#                 # if logger:
#                 #     logger.info(f"load_all_results: Completed loading file {file_count}/{len(pkl_files)}: {p.name}")

#                 # Group as E{cnn} only if H == fixed_trf_for_E (vary CNN while TRF fixed)
#                 if trf_dim == fixed_trf_for_E:
#                     key_E = f"E{cnn_dim}"
#                     entry["runs"].setdefault(key_E, []).append(rec)
#                     if logger:
#                         logger.debug(f"load_all_results: Added {p.name} to group {key_E}")

#                 # Group as H{trf} only if E == fixed_cnn_for_H (vary TRF while CNN fixed)
#                 if cnn_dim == fixed_cnn_for_H:
#                     key_H = f"H{trf_dim}"
#                     entry["runs"].setdefault(key_H, []).append(rec)
#                     if logger:
#                         logger.debug(f"load_all_results: Added {p.name} to group {key_H}")

#         # sort each dimension's list by run index
#         for k in list(entry["runs"].keys()):
#             entry["runs"][k].sort(key=lambda d: d["run"])
#         if logger:
#             runs_per_group = {k: len(v) for k, v in entry['runs'].items()}
#             logger.debug(f"load_all_results: Task '{task}' complete - groups: {list(entry['runs'].keys())}, runs per group: {runs_per_group}")

#         # attach class names if provided as a mapping
#         if classmap is not None and task in classmap:
#             entry["__class_names__"] = classmap[task]

#         out[task] = entry

#     if logger:
#         logger.debug(f"load_all_results: Completed loading all results. Total tasks: {len(out)}")
#     return out

def get_style(style = "stylesD"):
    
    with open(f'src/visualization/{style}.yaml', "r") as f:
        styles = yaml.safe_load(f)["styles"]

    # Task-specific class names
    TASK_CLASS_NAMES = {
        'MetabolicState_2': ["Starved", "Fed"],
        'State_Modality_6': [
            "Odor (S)", "Odor (F)", "Taste (S)", "Taste (F)",
            "Odor + Taste (S)", "Odor + Taste (F)"
        ],
        'State_Modality_Valence_16': [
            "O$^{+}$ (S)", "O$^{-}$ (S)", "O$^{+}$ (F)", "O$^{-}$ (F)",
            "T$^{+}$ (S)", "T$^{-}$ (S)", "T$^{+}$ (F)", "T$^{-}$ (F)",
            "O$^{+}$+T$^{+}$ (S)", "O$^{-}$+T$^{-}$ (S)", "O$^{-}$+T$^{+}$ (S)", "O$^{+}$+T$^{-}$ (S)",
            "O$^{+}$+T$^{+}$ (F)", "O$^{-}$+T$^{-}$ (F)", "O$^{-}$+T$^{+}$ (F)", "O$^{+}$+T$^{-}$ (F)"
        ]
    }
    TASK_COLORS = {
        'MetabolicState_2': [styles['starved']['color'], styles['fed']['color']],
        'State_Modality_6': [
            styles['starved_odor']['color'], styles['fed_odor']['color'],
            styles['starved_taste']['color'], styles['fed_taste']['color'],
            styles['starved_odor_taste']['color'], styles['fed_odor_taste']['color']],

        'State_Modality_Valence_16': [
            styles['starved_odor_positive']['color'], styles['starved_odor_negative']['color'],
            styles['fed_odor_positive']['color'], styles['fed_odor_negative']['color'],

            styles['starved_taste_positive']['color'], styles['starved_taste_negative']['color'],
            styles['fed_taste_positive']['color'], styles['fed_taste_negative']['color'],

            styles['starved_odor_pos_taste_pos']['color'], styles['starved_odor_neg_taste_neg']['color'],
            styles['starved_odor_neg_taste_pos']['color'], styles['starved_odor_pos_taste_neg']['color'],

            styles['fed_odor_pos_taste_pos']['color'], styles['fed_odor_neg_taste_neg']['color'],
            styles['fed_odor_neg_taste_pos']['color'], styles['fed_odor_pos_taste_neg']['color']]
    }

    TASK_EDGECOLORS = {
        'MetabolicState_2': [styles['starved']['edgecolor'], styles['fed']['edgecolor']],
        'State_Modality_6': [
            styles['starved_odor']['edgecolor'], styles['fed_odor']['edgecolor'],
            styles['starved_taste']['edgecolor'], styles['fed_taste']['edgecolor'],
            styles['starved_odor_taste']['edgecolor'], styles['fed_odor_taste']['edgecolor']],

        'State_Modality_Valence_16': [
            styles['starved_odor_positive']['edgecolor'], styles['starved_odor_negative']['edgecolor'],
            styles['fed_odor_positive']['edgecolor'], styles['fed_odor_negative']['edgecolor'],

            styles['starved_taste_positive']['edgecolor'], styles['starved_taste_negative']['edgecolor'],
            styles['fed_taste_positive']['edgecolor'], styles['fed_taste_negative']['edgecolor'],

            styles['starved_odor_pos_taste_pos']['edgecolor'], styles['starved_odor_neg_taste_neg']['edgecolor'],
            styles['starved_odor_neg_taste_pos']['edgecolor'], styles['starved_odor_pos_taste_neg']['edgecolor'],

            styles['fed_odor_pos_taste_pos']['edgecolor'], styles['fed_odor_neg_taste_neg']['edgecolor'],
            styles['fed_odor_neg_taste_pos']['edgecolor'], styles['fed_odor_pos_taste_neg']['edgecolor']
        ]
    }

    TASK_SHAPES = {
        'MetabolicState_2': [styles['starved']['shape'], styles['fed']['shape']],
        'State_Modality_6': [
            styles['starved_odor']['shape'], styles['starved_taste']['shape'],
            styles['starved_odor_taste']['shape'], styles['fed_odor']['shape'],
            styles['fed_taste']['shape'], styles['fed_odor_taste']['shape']
        ],
        'State_Modality_Valence_16': [
            styles['starved_odor_positive']['shape'], styles['starved_odor_negative']['shape'],
            styles['fed_odor_positive']['shape'], styles['fed_odor_negative']['shape'],

            styles['starved_taste_positive']['shape'], styles['starved_taste_negative']['shape'],
            styles['fed_taste_positive']['shape'], styles['fed_taste_negative']['shape'],

            styles['starved_odor_pos_taste_pos']['shape'], styles['starved_odor_neg_taste_neg']['shape'],
            styles['starved_odor_neg_taste_pos']['shape'], styles['starved_odor_pos_taste_neg']['shape'],
            styles['fed_odor_pos_taste_pos']['shape'], styles['fed_odor_neg_taste_neg']['shape'],
            styles['fed_odor_neg_taste_pos']['shape'], styles['fed_odor_pos_taste_neg']['shape']
        ]
    }

    return styles, TASK_CLASS_NAMES, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES






def load_all_results(
    base_dir: str | os.PathLike,
    task_names: Optional[Iterable[str] | Mapping[str, Any]] = None,
    fixed_trf_for_E: int = 16,   # H must equal this to populate E* groups
    fixed_cnn_for_H: int = 16,   # E must equal this to populate H* groups
    logger=None,
) -> Dict[str, Dict[str, Any]]:
    """
    Memory-safe loader.

    - For runs/dim_runs pickles: keeps ONLY 'accuracy' (plus metadata/path).
    - For control/best: keeps ONLY selected keys.
    - Groups runs into:
        E{cnn_dim} if trf_dim == fixed_trf_for_E
        H{trf_dim} if cnn_dim == fixed_cnn_for_H
    - Only strict filenames C{classes}_E{cnn}_H{trf}_{run}.pkl (or -run) are considered.
    """

    RUNS_KEEP = {"accuracy"}
    CONTROL_BEST_KEEP = {
        "accuracy",
        "confusion_matrix",
        "classification_report_dict",
        "transformer_latent_space",
        "latent_labels",
        "tsne_2d",
        "mean_cams",
    }

    base = Path(base_dir)

    strict = re.compile(
        r'^C(?P<classes>\d+)_E(?P<cnn>\d+)_H(?P<trf>\d+)[_-](?P<run>\d+)\.pkl$',
        re.IGNORECASE
    )

    def parse_filename(name: str):
        m = strict.match(name)
        if not m:
            raise ValueError(f"Non-matching filename '{name}'")
        return (int(m["classes"]), int(m["cnn"]), int(m["trf"]), int(m["run"]))

    def load_pickle_filtered(path: Path, keep: set[str]) -> Optional[dict]:
        """Load pickle and keep only selected keys (dict expected)."""
        try:
            with path.open("rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict):
                # unexpected format; return as-is (but this might be large)
                return None

            out = {k: data.get(k) for k in keep if k in data}

            # free ASAP
            del data
            gc.collect()
            return out
        except Exception as e:
            if logger:
                logger.warning(f"Failed to load {path}: {e}")
            else:
                print(f"[WARN] Failed to load {path}: {e}")
            return None

    def load_latest_pkl_filtered(dir_path: Path, keep: set[str], return_path: bool = False) -> Optional[dict]:
        """Load newest .pkl in dir_path, filtered to keys.
        
        If return_path=True, returns (data_dict, filename) tuple instead of just dict.
        """
        if not dir_path.is_dir():
            return (None, None) if return_path else None
        pkls = [p for p in dir_path.iterdir() if p.is_file() and p.suffix == ".pkl"]
        if not pkls:
            return (None, None) if return_path else None
        pkls.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        data = load_pickle_filtered(pkls[0], keep=keep)
        if return_path:
            return (data, pkls[0].name)
        return data

    # Normalize task_names into whitelist and optional class-name mapping
    whitelist: Optional[set[str]] = None
    classmap: Optional[Mapping[str, Any]] = None
    if task_names is not None:
        if isinstance(task_names, Mapping):
            classmap = task_names
            whitelist = set(task_names.keys())
        else:
            whitelist = set(task_names)

    # choose which task folders to walk
    tasks = [p for p in base.iterdir() if p.is_dir()]
    if whitelist is not None:
        tasks = [p for p in tasks if p.name in whitelist]
    tasks.sort(key=lambda p: p.name)

    out: Dict[str, Dict[str, Any]] = {}

    for task_dir in tasks:
        task = task_dir.name
        entry: Dict[str, Any] = {"control": None, "best": None, "runs": {}}

        # control (support 'control' OR 'control_run') - filtered
        for c in ("control", "control_run"):
            cand = task_dir / c
            v = load_latest_pkl_filtered(cand, keep=CONTROL_BEST_KEEP)
            if v is not None:
                entry["control"] = v
                break

        # best - filtered (also capture filename to extract dimension info)
        best_data, best_filename = load_latest_pkl_filtered(task_dir / "best", keep=CONTROL_BEST_KEEP, return_path=True)
        entry["best"] = best_data
        entry["best_filename"] = best_filename
        # Parse best filename to extract dimensions (e.g., C6E16_H8_3.pkl -> cnn=16, trf=8)
        entry["best_cnn_dim"] = None
        entry["best_trf_dim"] = None
        if best_filename:
            try:
                _, cnn_dim, trf_dim, _ = parse_filename(best_filename)
                entry["best_cnn_dim"] = cnn_dim
                entry["best_trf_dim"] = trf_dim
            except Exception:
                pass  # filename didn't match expected pattern

        # runs (support 'runs' OR 'dim_runs')
        for runs_folder in ("runs", "dim_runs"):
            rdir = task_dir / runs_folder
            if not rdir.is_dir():
                continue

            pkl_files = [p for p in rdir.iterdir() if p.is_file() and p.suffix == ".pkl"]
            for p in sorted(pkl_files, key=lambda x: x.name):
                try:
                    classes, cnn_dim, trf_dim, run = parse_filename(p.name)
                except Exception:
                    continue

                # load only accuracy for run pickles
                small = load_pickle_filtered(p, keep=RUNS_KEEP)
                if small is None:
                    continue

                rec = {
                    "run": run,
                    "classes": classes,
                    "cnn_dim": cnn_dim,
                    "trf_dim": trf_dim,
                    "path": str(p),
                    "accuracy": small.get("accuracy", None),
                }
                del small
                gc.collect()

                # group
                if trf_dim == fixed_trf_for_E:
                    entry["runs"].setdefault(f"E{cnn_dim}", []).append(rec)
                if cnn_dim == fixed_cnn_for_H:
                    entry["runs"].setdefault(f"H{trf_dim}", []).append(rec)

        # sort each group's list by run index
        for k in list(entry["runs"].keys()):
            entry["runs"][k].sort(key=lambda d: d["run"])

        # attach class names if provided as mapping
        if classmap is not None and task in classmap:
            entry["__class_names__"] = classmap[task]

        out[task] = entry

    return out


def load_h16_classification_reports(entry):
    """
    For all runs at d_model=16 (H16 group), load the full 
    classification_report_dict from disk.
    Returns a list of report dicts.
    """
    reports = []
    h16_runs = entry.get("runs", {}).get("H16", [])
    for run in h16_runs:
        path = run.get("path")
        if path is None:
            continue
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            report = data.get("classification_report_dict")
            if report is not None:
                reports.append(report)
            del data
            gc.collect()
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
    return reports
