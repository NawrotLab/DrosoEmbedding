import argparse
import os
import pickle
from pathlib import Path

from torch.utils.data import DataLoader
from pytorch_grad_cam import (
    AblationCAM,
    EigenCAM,
    FullGrad,
    GradCAM,
    GradCAMPlusPlus,
    HiResCAM,
    ScoreCAM,
    XGradCAM,
)

from src.data.dataset import CustomDataset
from src.models.cnn_transformer import CNN_Transformer
from src.models.model_io import load_model
from src.utils.config_loader import load_config
from src.utils.helpers import paths2neuropilpaths
from src.utils.logger import setup_logger
from src.visualization.visualize_preformance import compute_class_mean_cams, plot_mean_cams


CAM_METHODS = {
    "GradCAM": GradCAM,
    "HiResCAM": HiResCAM,
    "ScoreCAM": ScoreCAM,
    "GradCAMPlusPlus": GradCAMPlusPlus,
    "AblationCAM": AblationCAM,
    "XGradCAM": XGradCAM,
    "EigenCAM": EigenCAM,
    "FullGrad": FullGrad,
}


def _resolve_cam_settings(config: dict):
    cam_cfg = config.get("visualization", {}).get("cam", {})
    method_name = cam_cfg.get("method", "GradCAM")
    target_layer = int(cam_cfg.get("target_layer", 3))

    if method_name not in CAM_METHODS:
        raise ValueError(
            f"Unknown CAM method '{method_name}'. "
            f"Choose one of: {list(CAM_METHODS.keys())}"
        )
    return CAM_METHODS[method_name], method_name, target_layer


def _resolve_results_root(config: dict, run_id_override: str, slurm_id_override, logger):
    root = Path(config["paths"]["root"])
    task = config["data"]["task"]
    base_run_id = run_id_override or config["run_id"]
    base_results_root = Path(config["paths"]["results_root"])

    candidates = [base_results_root]

    slurm_env = os.getenv("SLURM_ARRAY_TASK_ID")
    slurm_id = slurm_id_override if slurm_id_override is not None else slurm_env
    if slurm_id is not None:
        candidates.append(root / "results" / f"{task}_{base_run_id}_{slurm_id}")

    pattern = f"{task}_{base_run_id}_*"
    discovered = [p for p in (root / "results").glob(pattern) if p.is_dir()]
    discovered.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    candidates.extend(discovered)

    seen = set()
    uniq_candidates = []
    for c in candidates:
        cs = str(c)
        if cs not in seen:
            uniq_candidates.append(c)
            seen.add(cs)

    for c in uniq_candidates:
        if (c / "config.pkl").exists():
            logger.info(f"Using results root: {c}")
            return c

    for c in uniq_candidates:
        if c.exists():
            logger.warning(
                f"No config.pkl found, but using existing results root: {c}"
            )
            return c

    logger.warning(
        f"No matching results directory found for task={task}, run_id={base_run_id}. "
        f"Falling back to config path: {base_results_root}"
    )
    return base_results_root


def main():
    parser = argparse.ArgumentParser(description="Compute and plot mean class Grad-CAMs.")
    parser.add_argument(
        "--max-items-per-class",
        type=int,
        default=None,
        help="Cap samples per class for CAM averaging (default: all samples).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config for CAM dataloader.",
    )
    parser.add_argument(
        "--model-subdir",
        type=str,
        default="best",
        help="Model subdirectory under config['paths']['models'] (default: best).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override run_id from config.yaml (without task prefix).",
    )
    parser.add_argument(
        "--slurm-id",
        type=int,
        default=None,
        help="Optional SLURM suffix for run folder (e.g. 42 -> ..._<run_id>_42).",
    )
    args = parser.parse_args()

    config = load_config()
    run_name = args.run_id or config["run_id"]
    logger = setup_logger(task_name=run_name, log_dir="logs/VizCAM")

    results_root = _resolve_results_root(config, args.run_id, args.slurm_id, logger)
    config_path = str(results_root / "config.pkl")
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config_train = pickle.load(f)
        logger.info(f"Loaded training config from {config_path}")
    else:
        config_train = config
        logger.warning(
            f"Training config not found at {config_path}. Falling back to current config.yaml."
        )

    model_params = config_train["model"]["parameters"]
    paths = config_train["paths"]
    batch_size = args.batch_size or config_train["training"]["batch_size"]

    with open(paths["pickle_path"], "rb") as f:
        _, _, X_test, _, _, Y_test = pickle.load(f)
    X_test = paths2neuropilpaths(X_test, config_train)

    test_dataset = CustomDataset(
        X_test,
        Y_test,
        transform=True,
        seq_length=model_params["seq_len"],
        seq_steps=model_params["seq_steps"],
        allTs_path=config_train["paths"]["allTs_path"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    logger.info(f"Loaded test dataset with {len(test_dataset)} samples.")

    model_dir = os.path.join(paths["models"], args.model_subdir)
    if not os.path.exists(model_dir):
        logger.warning(
            f"Model subdir not found: {model_dir}. Falling back to {paths['models']}."
        )
        model_dir = paths["models"]

    model, _, _, _, _, _ = load_model(
        CNN_Transformer,
        model_params,
        model_dir,
        config_train["device"],
        logger,
    )
    if model is None:
        raise FileNotFoundError(f"Trained model not found in {model_dir}")

    cam_method, cam_method_name, target_layer_index = _resolve_cam_settings(config_train)
    class_names = config_train["data"]["classes"]
    logger.info(
        f"Computing CAMs with method={cam_method_name}, target_layer={target_layer_index}, "
        f"max_items_per_class={args.max_items_per_class}"
    )

    mean_cams, _ = compute_class_mean_cams(
        model=model,
        dataloader=test_loader,
        class_names=class_names,
        device=config_train["device"],
        cam_method=cam_method,
        target_layer_index=target_layer_index,
        use_reshape_transform=True,
        max_items_per_class=args.max_items_per_class,
        return_all=False,
    )

    out_dir = os.path.join(paths["visualizations"], "cam")
    os.makedirs(out_dir, exist_ok=True)
    plot_mean_cams(
        mean_cams,
        cols=4,
        figsize=(10, 8),
        suptitle=f"Mean {cam_method_name} (Layer {target_layer_index})",
        add_colorbar=True,
        save_path=out_dir,
    )
    with open(os.path.join(out_dir, f"mean_cams_{cam_method_name}_L{target_layer_index}.pkl"), "wb") as f:
        pickle.dump(mean_cams, f)

    logger.info(f"Saved mean CAM plot and data to {out_dir}")


if __name__ == "__main__":
    main()
