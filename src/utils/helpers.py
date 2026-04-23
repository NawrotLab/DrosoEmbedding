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
from matplotlib.path import Path as MplPath

# ── Bicolor half-circle markers ──────────────────────────────────────
def _make_half_circle(side='left'):
    """Create a half-circle marker Path for bicolor split-circle rendering."""
    if side == 'left':
        theta = np.linspace(np.pi / 2, 3 * np.pi / 2, 50)
    else:
        theta = np.linspace(-np.pi / 2, np.pi / 2, 50)
    verts = np.column_stack([np.cos(theta), np.sin(theta)])
    verts = np.vstack([[0, 0], verts, [0, 0]])
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * len(theta) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)

HALF_CIRCLE_LEFT = _make_half_circle('left')
HALF_CIRCLE_RIGHT = _make_half_circle('right')


def scatter_bicolor(ax, x, y, style, s=120, linewidth=0.5, zorder=3, **kwargs):
    """
    Draw a bicolor split-circle at (x, y).
    `style` must be a dict with keys: left_color, right_color,
    left_edgecolor, right_edgecolor.
    Falls back to a regular scatter if bicolor info is missing.
    """
    ax.scatter(x, y, marker=HALF_CIRCLE_LEFT,
               c=[style['left_color']], edgecolors=[style['left_edgecolor']],
               s=s, linewidth=linewidth, zorder=zorder, **kwargs)
    ax.scatter(x, y, marker=HALF_CIRCLE_RIGHT,
               c=[style['right_color']], edgecolors=[style['right_edgecolor']],
               s=s, linewidth=linewidth, zorder=zorder, **kwargs)


def scatter_bicolor_cloud(ax, xs, ys, style, s=6, alpha=0.8, linewidth=0.3, zorder=1):
    """
    Draw a cloud of bicolor split-circle points (e.g. for t-SNE).
    """
    ax.scatter(xs, ys, marker=HALF_CIRCLE_LEFT,
               c=style['left_color'], edgecolors=style['left_edgecolor'],
               s=s, alpha=alpha, linewidth=linewidth, zorder=zorder)
    ax.scatter(xs, ys, marker=HALF_CIRCLE_RIGHT,
               c=style['right_color'], edgecolors=style['right_edgecolor'],
               s=s, alpha=alpha, linewidth=linewidth, zorder=zorder)


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



def get_group_map(task_name: str, class_names: list[str]) -> dict[str, list[int]]:
    """
    Return the group mapping for the given task and class names.
    
    Canonical version — used by all figure scripts and visualize_preformance.
    """
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

def get_style(style = "styles"):
    
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

    # ── Bicolor info for split-circle rendering ──
    # Maps task → { class_index: { left_color, right_color, left_edgecolor, right_edgecolor } }
    def _bicolor_entry(style_key):
        s = styles[style_key]
        if not s.get('bicolor', False):
            return None
        return {k: s[k] for k in ('left_color', 'right_color', 'left_edgecolor', 'right_edgecolor')}

    # 16-class list order matches TASK_COLORS ordering above:
    #  0: starved_odor_positive        8:  starved_odor_pos_taste_pos
    #  1: starved_odor_negative        9:  starved_odor_neg_taste_neg
    #  2: fed_odor_positive            10: starved_odor_neg_taste_pos  ← bicolor
    #  3: fed_odor_negative            11: starved_odor_pos_taste_neg  ← bicolor
    #  4: starved_taste_positive       12: fed_odor_pos_taste_pos
    #  5: starved_taste_negative       13: fed_odor_neg_taste_neg
    #  6: fed_taste_positive           14: fed_odor_neg_taste_pos      ← bicolor
    #  7: fed_taste_negative           15: fed_odor_pos_taste_neg      ← bicolor
    _keys_16 = [
        'starved_odor_positive', 'starved_odor_negative',
        'fed_odor_positive', 'fed_odor_negative',
        'starved_taste_positive', 'starved_taste_negative',
        'fed_taste_positive', 'fed_taste_negative',
        'starved_odor_pos_taste_pos', 'starved_odor_neg_taste_neg',
        'starved_odor_neg_taste_pos', 'starved_odor_pos_taste_neg',
        'fed_odor_pos_taste_pos', 'fed_odor_neg_taste_neg',
        'fed_odor_neg_taste_pos', 'fed_odor_pos_taste_neg',
    ]
    bicolor_16 = {}
    for idx, key in enumerate(_keys_16):
        entry = _bicolor_entry(key)
        if entry is not None:
            bicolor_16[idx] = entry

    TASK_BICOLOR_INFO = {
        'MetabolicState_2': {},
        'State_Modality_6': {},
        'State_Modality_Valence_16': bicolor_16,
    }

    return styles, TASK_CLASS_NAMES, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES, TASK_BICOLOR_INFO






def load_all_results(
    base_dir: str | os.PathLike,
    task_names: Optional[Iterable[str] | Mapping[str, Any]] = None,
    fixed_trf_for_E: int = 16,   # H must equal this to populate E* groups
    fixed_cnn_for_H: int = 16,   # E must equal this to populate H* groups
    only_cnn_dim: Optional[int] = None,  # if set, skip runs where cnn_dim != this value
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

                # Skip files not matching required CNN dimension
                if only_cnn_dim is not None and cnn_dim != only_cnn_dim:
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