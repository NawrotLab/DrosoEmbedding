from __future__ import annotations
from pathlib import Path
from typing import Iterable, Mapping, Any, Dict, Optional
import os, re, pickle
import numpy as np
import torch
from sklearn.manifold import TSNE
import yaml
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


def paths2neuropilpaths(X, config):
    prep = config["data"]["preprocessing"]

    if not prep["isolate_neuropil"]:
        return X  # nothing to do

    X_new = []
    for p in X:
        p = Path(p)
        parts = list(p.parts)

        # accept either naming: older pickles' paths say "meanZ_logTs"
        # (a directory that no longer gets created), newer ones say
        # "meanZ_allTs" -- clean_dataset.py only ever writes the latter now.
        base = next((b for b in ("meanZ_logTs", "meanZ_allTs") if b in parts), None)
        if base is None:
            raise ValueError(f"Expected 'meanZ_logTs' or 'meanZ_allTs' in path but got:\n{p}")

        parts[parts.index(base)] = f"{base}_{prep['neuropil']}"
        X_new.append(str(Path(*parts)))

    return X_new


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
            outputs = model(sequences)
            predicted = torch.argmax(outputs, dim=1)
            predictions.append(predicted.cpu().numpy())
            true_labels.append(labels.cpu().numpy())
    predictions_np = np.concatenate(predictions, axis=0)
    true_labels_np = np.concatenate(true_labels, axis=0)
    return predictions_np, true_labels_np


def get_predictions_with_probs(model, data_loader, device):
    """Like get_predictions but also returns softmax probabilities (N, n_classes)."""
    import torch.nn.functional as F
    model.eval()
    predictions, true_labels, all_probs = [], [], []
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences = sequences.float().to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            probs = F.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1)
            predictions.append(predicted.cpu().numpy())
            true_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    return (np.concatenate(predictions),
            np.concatenate(true_labels),
            np.concatenate(all_probs)) 

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
    
    Canonical version — used by all figure scripts and visualize_performance.
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






# The one "best" model per task -- used for confusion matrices, t-SNE and
# interpretability figures -- is chosen by load_all_results(): the
# non-control run at the canonical architecture (cnn_dim=fixed_cnn_for_H,
# trf_dim=fixed_trf_for_E, i.e. E16_H16) with the highest best-epoch
# validation accuracy (max of the saved val_acc history; ties -> lowest
# run index). Validation, not test accuracy, so the test set doesn't
# leak into model selection.
_RUN_ID_RE = re.compile(
    r'^C(?P<classes>\d+)(?:_(?P<ctrl>Ctr))?_E(?P<cnn>\d+)(?:_H(?P<trf>\d+))?_(?P<run>\d+)$',
    re.IGNORECASE
)


def parse_run_id(run_id: str):
    """(classes, cnn_dim, trf_dim, run, is_control) from a run_id string,
    e.g. 'C16_E16_H16_10' or the control form 'C16_Ctr_E16_0' (which has
    no H-component -- trf_dim comes back None for those). None if the
    run_id doesn't match the expected pattern at all."""
    m = _RUN_ID_RE.match(run_id)
    if not m:
        return None
    classes = int(m['classes'])
    is_control = m['ctrl'] is not None
    cnn = int(m['cnn'])
    trf = int(m['trf']) if m['trf'] is not None else None
    run = int(m['run'])
    return classes, cnn, trf, run, is_control


# Layout version of the aggregated results (what load_all_results() returns and
# load_or_build_all_results() stores). Bump when entries gain/lose fields, so an
# older file on disk is recognised as outdated instead of silently misused.
#   2: canonical runs carry confusion_matrix, classification_report_dict, curves
RESULTS_VERSION = 2


def load_all_results(
    base_dir: str | os.PathLike,
    task_names: Optional[Iterable[str] | Mapping[str, Any]] = None,
    fixed_trf_for_E: int = 16,   # H must equal this to populate E* groups
    fixed_cnn_for_H: int = 16,   # E must equal this to populate H* groups
    only_cnn_dim: Optional[int] = None,  # if set, skip runs where cnn_dim != this value
    logger=None,
) -> Dict[str, Dict[str, Any]]:
    """
    Memory-safe loader for the published evaluation/<task>/{run_id}_evalResults.pkl
    layout (flat, one file per model -- run_evaluation.py's output). Replaces
    the old _chkpt_finals/<task>/{control,best,runs}/ cluster-only convention
    -- same grouping semantics, different source layout.

    - For regular runs: keeps ONLY 'accuracy' and the best-epoch validation
      accuracy 'val_acc_best' (memory-safe).
    - For control/the "best" run: keeps the full result (confusion matrix,
      latent space, t-SNE, CAMs, etc). "Best" = the non-control run with
      cnn_dim == fixed_cnn_for_H and trf_dim == fixed_trf_for_E (E16_H16
      by default) with the highest val_acc_best; ties go to the lowest run
      index. entry["best_run_id"] names it, and entry["best_ranking"]
      lists the top 5 candidates (run_id, val_acc_best, test accuracy) so
      the selection can be sanity-checked.
    - Groups runs into:
        E{cnn_dim} if trf_dim == fixed_trf_for_E
        H{trf_dim} if cnn_dim == fixed_cnn_for_H
    - Only filenames matching parse_run_id() are considered; anything else
      is skipped with a warning.
    """

    RUNS_KEEP = {"accuracy", "val_acc"}
    # Canonical-architecture runs (E16_H16 by default) also carry everything the
    # sweep figures read per run -- confusion matrix + report (Fig 3) and the
    # loss/accuracy curves (Fig S3) -- so those figures need no per-model pkls.
    CANON_KEEP = RUNS_KEEP | {"confusion_matrix", "classification_report_dict", "train_loss", "val_loss"}
    CURVE_KEYS = ("train_loss", "val_loss", "val_acc")
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
        best_candidates = []  # (run_id, path, rec) for the canonical-architecture runs

        pkl_files = sorted(task_dir.glob('*_evalResults.pkl'))
        for p in pkl_files:
            run_id = p.name[:-len('_evalResults.pkl')]
            parsed = parse_run_id(run_id)
            if parsed is None:
                if logger:
                    logger.warning(f"Unrecognized filename, skipping: {p.name}")
                continue
            classes, cnn_dim, trf_dim, run, is_control = parsed

            if only_cnn_dim is not None and cnn_dim != only_cnn_dim:
                continue

            if is_control:
                entry["control"] = load_pickle_filtered(p, keep=CONTROL_BEST_KEEP)
                continue

            if trf_dim is None:
                continue  # not part of the E*/H* sweep grouping (e.g. malformed)

            is_canon = cnn_dim == fixed_cnn_for_H and trf_dim == fixed_trf_for_E
            small = load_pickle_filtered(p, keep=CANON_KEEP if is_canon else RUNS_KEEP)
            if small is None:
                continue

            val_hist = small.get("val_acc")
            rec = {
                "run": run,
                "classes": classes,
                "cnn_dim": cnn_dim,
                "trf_dim": trf_dim,
                "path": str(p),
                "accuracy": small.get("accuracy", None),
                "val_acc_best": float(np.max(val_hist)) if val_hist is not None and len(val_hist) else None,
            }
            if is_canon:
                cm = small.get("confusion_matrix")
                rec["confusion_matrix"] = np.asarray(cm) if cm is not None else None
                rec["classification_report_dict"] = small.get("classification_report_dict")
                # float64 on purpose: the curves are plotted as-is (mean/std over runs),
                # and the figure must come out identical to one built from the pkls.
                rec["curves"] = {k: np.asarray(small[k], dtype=float)
                                 for k in CURVE_KEYS if small.get(k) is not None}
            del small, val_hist
            gc.collect()

            # the "best" model is also a regular sweep member -- it stays in
            # the accuracy-only grouping below as well.
            if is_canon:
                if rec["val_acc_best"] is None:
                    if logger:
                        logger.warning(f"No val_acc history, not eligible as best: {p.name}")
                else:
                    best_candidates.append((run_id, p, rec))

            if trf_dim == fixed_trf_for_E:
                entry["runs"].setdefault(f"E{cnn_dim}", []).append(rec)
            if cnn_dim == fixed_cnn_for_H:
                entry["runs"].setdefault(f"H{trf_dim}", []).append(rec)

        # sort each group's list by run index
        for k in list(entry["runs"].keys()):
            entry["runs"][k].sort(key=lambda d: d["run"])

        entry["__version__"] = RESULTS_VERSION

        # pick the best: highest val_acc_best, ties -> lowest run index
        entry["best_run_id"] = None
        entry["best_filename"] = None
        entry["best_cnn_dim"] = None
        entry["best_trf_dim"] = None
        entry["best_ranking"] = []
        if best_candidates:
            best_candidates.sort(key=lambda c: (-c[2]["val_acc_best"], c[2]["run"]))
            entry["best_ranking"] = [
                {"run_id": rid, "val_acc_best": r["val_acc_best"], "accuracy": r["accuracy"]}
                for rid, _, r in best_candidates[:5]
            ]
            best_id, best_path, best_rec = best_candidates[0]
            entry["best"] = load_pickle_filtered(best_path, keep=CONTROL_BEST_KEEP)
            entry["best_run_id"] = best_id
            entry["best_filename"] = best_path.name
            entry["best_cnn_dim"] = best_rec["cnn_dim"]
            entry["best_trf_dim"] = best_rec["trf_dim"]
            if logger:
                top = ", ".join(f"{d['run_id']} ({d['val_acc_best']:.4f})" for d in entry["best_ranking"])
                logger.info(f"[{task}] best = {best_id} by val_acc_best; top 5: {top}")
        elif logger:
            logger.warning(f"[{task}] no E{fixed_cnn_for_H}_H{fixed_trf_for_E} candidates -- no best model selected")

        # attach class names if provided as mapping
        if classmap is not None and task in classmap:
            entry["__class_names__"] = classmap[task]

        out[task] = entry

    return out


def load_or_build_all_results(
    cache_path: str,
    base_dir: str | os.PathLike,
    task_names: Optional[Iterable[str] | Mapping[str, Any]] = None,
    fixed_trf_for_E: int = 16,
    fixed_cnn_for_H: int = 16,
    only_cnn_dim: Optional[int] = None,
    recompute: bool = False,
    logger=None,
) -> Dict[str, Dict[str, Any]]:
    """load_all_results(), cached to disk -- same pattern as
    load_or_run_ko_permutation(): every figure/analysis script that reads
    the full sweep would otherwise re-open and re-aggregate all ~750
    per-model pkls from scratch on every run, which is slow and entirely
    redundant once the sweep is stable. The cache is the *combined*
    across-all-tasks dict load_all_results() returns, at one location, so
    every caller shares the same cache regardless of which task(s) it asks
    for -- cheap either way, since the underlying data is already tiny
    (control/best keep the full rich result, ~1.9MB each; every other run
    keeps only 'accuracy').

    Pass recompute=True (or delete cache_path) to force a fresh build --
    there's no mtime/staleness check against the per-model pkls, so if any
    of them changed (e.g. a resubmitted sweep straggler), rebuild
    explicitly rather than relying on this to notice.
    """
    def _log(msg):
        if logger: logger.info(msg)
        else: print(msg)

    if not recompute and os.path.exists(cache_path):
        _log(f'Loading cached aggregated results ({cache_path})')
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        if all(isinstance(e, dict) and e.get('__version__', 0) >= RESULTS_VERSION for e in cached.values()):
            return cached
        _log(f'Cached results are outdated (need layout version {RESULTS_VERSION}) -- rebuilding')
        if not any(Path(base_dir).glob('*/*_evalResults.pkl')):
            raise RuntimeError(
                f'{cache_path} is an outdated results file (layout version < {RESULTS_VERSION}) and '
                f'there are no per-model evaluation pkls under {base_dir} to rebuild it from. '
                f'Get an up-to-date results file.')

    _log('Cache not found or recompute=True -- aggregating all results from disk…')
    results = load_all_results(
        base_dir, task_names,
        fixed_trf_for_E=fixed_trf_for_E, fixed_cnn_for_H=fixed_cnn_for_H,
        only_cnn_dim=only_cnn_dim, logger=logger,
    )
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)
    _log(f'Saved aggregated results -> {cache_path}')
    return results


def load_h16_classification_reports(entry):
    """
    For all runs at d_model=16 (H16 group), return the per-run
    classification_report_dict stored in the aggregated results
    (see load_all_results()) -- no per-model pkls are opened.
    Returns a list of report dicts.
    """
    reports = []
    for run in entry.get("runs", {}).get("H16", []):
        report = run.get("classification_report_dict")
        if report is not None:
            reports.append(report)
    return reports


def load_h16_reports_and_cms(entry, n_classes, logger=None):
    """Per-run classification reports and confusion matrices of the H16 group,
    read from the aggregated results (no per-model pkls are opened).

    Returns (reports, cms). Only call for tasks that actually need confusion
    matrices; for tasks that only need reports, use
    load_h16_classification_reports instead.
    """
    reports, cms = [], []
    for run in entry.get('runs', {}).get('H16', []):
        rpt = run.get('classification_report_dict')
        cm = run.get('confusion_matrix')
        if rpt is not None:
            reports.append(rpt)
        if cm is not None and np.array(cm).shape == (n_classes, n_classes):
            cms.append(np.array(cm))
    return reports, cms
