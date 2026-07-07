"""
Interpretability: Computation + Plotting
==========================================
GradCAM computation (via pytorch-grad-cam) and all plotting functions for Figure 4.

Changes vs. original
---------------------
- plot_gradcam_pooled:  `normalize_per_row=True` normalises each row independently
- save_gradcam_examples: saves N random input/attribution pairs to disk
- print_element_sizes:  reports sizes of every data artefact used in the figure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpecFromSubplotSpec
import matplotlib.image as mpimg
import torch
import torch.nn.functional as F
import os
import json
import cairosvg
from PIL import Image
import io
import matplotlib.transforms as mtransforms
import tifffile
from torchvision import transforms
from src.visualization.figure_base import apply_style, FONT_SIZES

apply_style()


# ════════════════════════════════════════════════
# GradCAM computation (using pytorch-grad-cam)
# ════════════════════════════════════════════════

def add_panel_label(fig, ax, label, dx=0.0, dy=0.01):
    bbox = ax.get_position()
    fig.text(
        bbox.x0 + dx,
        bbox.y1 + dy,
        label,
        ha='left',
        va='bottom',
        fontsize=FONT_SIZES['panel_label'],
        fontweight='bold'
    )

def load_image(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".svg":
        png_bytes = cairosvg.svg2png(url=path)
        img = Image.open(io.BytesIO(png_bytes))
        return np.array(img)
    else:
        return plt.imread(path)

def compute_gradcam(
    model, test_loader, target_layer, n_classes,
    device='cuda', input_size=128, cam_method=None, logger=None,
):
    """Compute per-class mean GradCAM heatmaps using pytorch-grad-cam.

    Also collects per-sample (input, cam) pairs for later example-saving.
    Returns
    -------
    mean_cams : ndarray (n_classes, H, W)
    all_examples : list[dict]   – each dict has keys 'input', 'cam', 'label'
    """
    from pytorch_grad_cam import GradCAMPlusPlus, GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    if cam_method is None:
        cam_method = GradCAMPlusPlus

    def _log(msg):
        if logger: logger.info(msg)
        else: print(msg)

    model.eval()
    first_batch = next(iter(test_loader))[0]
    seq_len = first_batch.shape[1] if first_batch.dim() == 5 else 1
    _log(f"  Sequence length: {seq_len}")

    cam = cam_method(model=model, target_layers=[target_layer])
    class_cams = {c: [] for c in range(n_classes)}
    all_examples = []                       # ← NEW: collect individual examples

    for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
        X_batch = X_batch.to(device)
        y_np = y_batch.numpy()

        if X_batch.dim() == 5:
            B, S, C, H, W = X_batch.shape
            frames = X_batch.reshape(B * S, C, H, W)
            y_expanded = np.repeat(y_np, S)
        else:
            frames = X_batch
            y_expanded = y_np
            S = 1

        for c in range(n_classes):
            mask = y_expanded == c
            if not mask.any():
                continue
            X_sub = frames[mask]
            targets = [ClassifierOutputTarget(c)] * X_sub.shape[0]
            grayscale_cams = cam(input_tensor=X_sub, targets=targets)

            # ── store individual examples ──────────────────────
            X_sub_np = X_sub.detach().cpu().numpy()
            for i in range(X_sub_np.shape[0]):
                gc = grayscale_cams[i]
                # upsample cam to input_size if needed
                if gc.shape[0] != input_size:
                    t = torch.tensor(gc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    gc = F.interpolate(t, size=(input_size, input_size),
                                       mode='bilinear', align_corners=False).squeeze().numpy()
                all_examples.append({
                    'input': X_sub_np[i],     # (C, H, W)
                    'cam':   gc,               # (H, W)
                    'label': int(c),
                })
            # ── end individual examples ────────────────────────

            if seq_len > 1:
                n_samples = mask.sum() // seq_len
                grayscale_cams = grayscale_cams.reshape(
                    n_samples, seq_len, grayscale_cams.shape[-2], grayscale_cams.shape[-1]
                ).mean(axis=1)

            class_cams[c].append(grayscale_cams)

        if batch_idx % 10 == 0:
            _log(f"  GradCAM batch {batch_idx}/{len(test_loader)}")

    mean_cams = []
    for c in range(n_classes):
        if class_cams[c]:
            mean_cam = np.concatenate(class_cams[c], axis=0).mean(axis=0)
        else:
            _log(f"  Warning: no samples for class {c}")
            mean_cam = np.zeros((input_size, input_size))

        if mean_cam.shape[0] != input_size:
            t = torch.tensor(mean_cam, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            mean_cam = F.interpolate(t, size=(input_size, input_size),
                                      mode='bilinear', align_corners=False).squeeze().numpy()
        mean_cams.append(mean_cam)

    mean_cams = np.stack(mean_cams)
    _log(f"  GradCAM done: {mean_cams.shape}")
    _log(f"  Collected {len(all_examples)} individual examples")
    del cam
    return mean_cams, all_examples


# ════════════════════════════════════════════════
# Save N random GradCAM examples
# ════════════════════════════════════════════════

def save_gradcam_examples(
    all_examples, out_dir, n=20, class_names=None,
    cmap='viridis', seed=42, logger=None,
):
    """Save *n* random (input, GradCAM attribution) side-by-side PNGs.

    Parameters
    ----------
    all_examples : list[dict]  – from compute_gradcam
    out_dir : str              – directory to write into
    n : int                    – number of examples
    class_names : list[str]    – optional pretty class labels

    Returns
    -------
    selected : list[dict]  – the *n* examples that were saved (same order),
        so downstream code can reuse the exact same samples.
    """
    def _log(msg):
        if logger: logger.info(msg)
        else: print(msg)

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(all_examples), size=min(n, len(all_examples)), replace=False)
    indices.sort()

    selected = []
    for rank, idx in enumerate(indices):
        ex = all_examples[idx]
        selected.append(ex)
        inp = ex['input']                # (C, H, W) or (1, H, W)
        cam_map = ex['cam']              # (H, W)
        label = ex['label']

        # Prepare the input image for display
        if inp.shape[0] == 1:
            img = inp[0]                 # grayscale
        elif inp.shape[0] == 3:
            img = np.transpose(inp, (1, 2, 0))  # RGB
        else:
            img = inp.mean(axis=0)       # average channels

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))

        axes[0].imshow(img, cmap='gray' if img.ndim == 2 else None, aspect='equal')
        axes[0].set_title('Input', fontsize=FONT_SIZES['subplot_title'])
        axes[0].axis('off')

        axes[1].imshow(cam_map, cmap=cmap, aspect='equal', vmin=0, vmax=cam_map.max())
        axes[1].set_title('Attribution', fontsize=FONT_SIZES['subplot_title'])
        axes[1].axis('off')

        cls_label = class_names[label] if class_names else str(label)
        fig.suptitle(f'Example {rank+1}  —  class: {cls_label}', fontsize=FONT_SIZES['title'])
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        fname = os.path.join(out_dir, f"gradcam_example_{rank+1:02d}_cls{label}.png")
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)

    _log(f"  Saved {len(selected)} GradCAM examples → {out_dir}")
    return selected


# ════════════════════════════════════════════════
# Save feature maps from a target layer
# ════════════════════════════════════════════════

def save_feature_maps(
    model, target_layer, selected_examples, out_dir,
    class_names=None, cmap='viridis', device='cuda', logger=None,
):
    """Extract and save feature-map activations from *target_layer*.

    Uses the **same samples** already chosen by ``save_gradcam_examples``
    so that input → feature map → attribution can be compared side-by-side.

    For each sample the function saves:
      • one grid overview  (all channels in a single figure)
      • one PNG per channel (individual feature maps)

    Parameters
    ----------
    model             : nn.Module
    target_layer      : nn.Module  – the Conv2d (or similar) layer to hook
    selected_examples : list[dict] – returned by save_gradcam_examples;
                        each dict has keys 'input' (C,H,W), 'cam', 'label'
    out_dir           : str        – output directory
    class_names       : list[str]  – optional pretty labels
    """
    def _log(msg):
        if logger: logger.info(msg)
        else: print(msg)

    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    # ── Forward hook to capture activations ────────────────────
    activations = {}

    def _hook_fn(_module, _input, output):
        activations['feat'] = output.detach().cpu()

    hook = target_layer.register_forward_hook(_hook_fn)

    n_channels = fh = fw = 0          # will be set on first sample

    for rank, ex in enumerate(selected_examples):
        inp_np = ex['input']           # (C, H, W)
        label = ex['label']

        img_tensor = torch.tensor(inp_np, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            _ = model(img_tensor)

        feat = activations['feat'].squeeze(0).numpy()   # (C_feat, H_feat, W_feat)
        n_channels, fh, fw = feat.shape
        cls_label = class_names[label] if class_names else str(label)

        # ── Prepare input image for display ────────────────────
        if inp_np.shape[0] == 1:
            inp_img = inp_np[0]
        elif inp_np.shape[0] == 3:
            inp_img = np.transpose(inp_np, (1, 2, 0))
        else:
            inp_img = inp_np.mean(axis=0)

        # ── Save individual channel PNGs ───────────────────────
        sample_dir = os.path.join(out_dir, f"sample_{rank+1:02d}_cls{label}")
        os.makedirs(sample_dir, exist_ok=True)

        for ch in range(n_channels):
            fig_ch, ax_ch = plt.subplots(1, 1, figsize=(2, 2))
            ax_ch.imshow(feat[ch], cmap=cmap, aspect='equal')
            ax_ch.axis('off')
            fig_ch.savefig(
                os.path.join(sample_dir, f"ch{ch:03d}.png"),
                dpi=100, bbox_inches='tight', pad_inches=0.02,
            )
            plt.close(fig_ch)

        # ── Save overview grid: input + all channels ───────────
        ncols = min(8, n_channels + 1)
        nrows = int(np.ceil((n_channels + 1) / ncols))
        fig_grid, axes_grid = plt.subplots(
            nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows),
        )
        axes_flat = np.array(axes_grid).flatten()

        # First cell: input image
        axes_flat[0].imshow(inp_img, cmap='gray' if inp_img.ndim == 2 else None, aspect='equal')
        axes_flat[0].set_title('Input', fontsize=FONT_SIZES['subplot_title'])
        axes_flat[0].axis('off')

        # Remaining cells: feature map channels
        for ch in range(n_channels):
            ax = axes_flat[ch + 1]
            ax.imshow(feat[ch], cmap=cmap, aspect='equal')
            ax.set_title(f'ch {ch}', fontsize=FONT_SIZES['subplot_title'])
            ax.axis('off')

        # Hide unused cells
        for j in range(n_channels + 1, len(axes_flat)):
            axes_flat[j].axis('off')

        fig_grid.suptitle(
            f'Sample {rank+1} — class: {cls_label}  |  '
            f'layer shape: ({n_channels}, {fh}, {fw})',
            fontsize=FONT_SIZES['title'],
        )
        fig_grid.tight_layout(rect=[0, 0, 1, 0.95])
        fig_grid.savefig(
            os.path.join(out_dir, f"featuremaps_{rank+1:02d}_cls{label}_grid.png"),
            dpi=150, bbox_inches='tight',
        )
        plt.close(fig_grid)

    hook.remove()
    _log(f"  Saved feature maps for {len(selected_examples)} samples → {out_dir}")
    if n_channels:
        _log(f"    Layer output shape: ({n_channels}, {fh}, {fw})  "
             f"= {n_channels} channels of {fh}×{fw} pixels")


# ════════════════════════════════════════════════
# GradCAM pooling
# ════════════════════════════════════════════════

# Factor definitions: name → list of class indices
FACTOR_POOLS = {
    'Fed':       [2, 3, 6, 7, 12, 13, 14, 15],
    'Starved':   [0, 1, 4, 5, 8, 9, 10, 11],
    'Odor':      [0, 1, 2, 3],
    'Taste':     [4, 5, 6, 7],
    'Odor+Taste':[8, 9, 10, 11, 12, 13, 14, 15],
    'App.':      [0, 2, 4, 6, 8, 12],
    'Avers.':    [1, 3, 5, 7, 9, 13],
    'Mixed':     [10, 11, 14, 15],
}


def pool_gradcams(mean_cams, factor_pools=None):
    """Average per-class GradCAMs into factor-level maps. Returns dict name→(H,W)."""
    if factor_pools is None:
        factor_pools = FACTOR_POOLS
    pooled = {}
    for name, indices in factor_pools.items():
        avg = mean_cams[indices].mean(axis=0)
        # Normalize to [0,1]
        cmin, cmax = avg.min(), avg.max()
        if cmax - cmin > 1e-8:
            avg = (avg - cmin) / (cmax - cmin)
        pooled[name] = avg
    return pooled


# ════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════

def load_neuropil_data(data_dir):
    """Load precomputed neuropil importance data."""
    data = {}
    data['W_abs'] = np.load(os.path.join(data_dir, "W_abs.npy"))
    data['W_signed'] = np.load(os.path.join(data_dir, "W_signed.npy"))
    with open(os.path.join(data_dir, "neuropil_names.json"), 'r') as f:
        data['neuropil_names'] = json.load(f)
    with open(os.path.join(data_dir, "ridge_summary.json"), 'r') as f:
        data['ridge_summary'] = json.load(f)
    for key, fname in [
        ('group_profiles_abs', 'group_profiles_abs.csv'),
        ('group_contrasts_abs', 'group_contrasts_abs.csv'),
        ('correlation', 'neuropil_correlation.csv'),
    ]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            data[key] = pd.read_csv(path, index_col=0)
    return data


# ════════════════════════════════════════════════
# Print element sizes
# ════════════════════════════════════════════════

def print_element_sizes(
    mean_cams, pooled_cams, np_data, class_names,
    sketch_path=None, atlas_path=None, gradcam_sketch_path=None,
):
    """Print size / shape / resolution of every element used in Figure 4."""
    sep = "─" * 60
    print(f"\n{sep}")
    print("ELEMENT SIZES — Figure 4: Interpretability")
    print(sep)

    # 1. Raw GradCAM
    print(f"\n[mean_cams]")
    print(f"  shape            : {mean_cams.shape}")
    print(f"  dtype            : {mean_cams.dtype}")
    print(f"  n_classes        : {mean_cams.shape[0]}")
    print(f"  spatial (H × W)  : {mean_cams.shape[1]} × {mean_cams.shape[2]}")
    print(f"  value range      : [{mean_cams.min():.4f}, {mean_cams.max():.4f}]")

    # 2. Pooled GradCAM
    print(f"\n[pooled_cams]  (n_factors = {len(pooled_cams)})")
    for name, arr in pooled_cams.items():
        print(f"  {name:15s}  shape={arr.shape}  range=[{arr.min():.3f}, {arr.max():.3f}]")

    # 3. Neuropil data
    print(f"\n[neuropil_names]   len = {len(np_data['neuropil_names'])}")
    print(f"  names: {np_data['neuropil_names']}")

    if 'W_abs' in np_data:
        W = np_data['W_abs']
        print(f"\n[W_abs]")
        print(f"  shape  : {W.shape}   (= n_neuropils × n_classes)")
        print(f"  formula: |W_ridge|  — absolute ridge regression weights")

    if 'group_profiles_abs' in np_data:
        gp = np_data['group_profiles_abs']
        print(f"\n[group_profiles_abs]  (heatmap data)")
        print(f"  shape  : {gp.shape}  (groups × neuropils)")
        print(f"  groups : {list(gp.index)}")
        print(f"  cols   : {list(gp.columns)}")
        print(f"  formula: mean(|W_ridge|) over classes within each factor group")

    if 'group_contrasts_abs' in np_data:
        gc = np_data['group_contrasts_abs']
        print(f"\n[group_contrasts_abs]  (contrast bar data)")
        print(f"  shape     : {gc.shape}  (contrasts × neuropils)")
        print(f"  contrasts : {list(gc.index)}")
        print(f"  formula   : mean_groupA(|W|) − mean_groupB(|W|)")
        print(f"              e.g. Starved_minus_Fed = mean_starved(|W|) - mean_fed(|W|)")

    # 4. Class names
    print(f"\n[class_names]   n = {len(class_names)}")
    print(f"  {class_names}")

    # 5. Static images
    for label, path in [
        ("GradCAM sketch", gradcam_sketch_path),
        ("Ridge sketch",   sketch_path),
        ("Neuropil atlas", atlas_path),
    ]:
        print(f"\n[{label}]")
        # if path and os.path.exists(path):
        #     # img = mpimg.imread(path)
        #     img = plt.imread(path)
        #     print(f"  path   : {path}")
        #     print(f"  shape  : {img.shape}  (H × W × channels)")
        #     print(f"  dtype  : {img.dtype}")
        #     print(f"  pixels : {img.shape[0]} × {img.shape[1]}")
        # else:
        #     print(f"  path   : {path}  — NOT FOUND")

        ext = os.path.splitext(path)[-1].lower()

        if ext == ".svg":
            print(f"  path   : {path}")
            print(f"  format : SVG (vector)")
            print(f"  shape  : resolution-independent")
        else:
            # img = plt.imread(path)
            img = load_image(path)
            print(f"  path   : {path}")
            # print(f"  shape  : {img.shape}  (H × W × channels)")
            # print(f"  dtype  : {img.dtype}")
            # print(f"  pixels : {img.shape[0]} × {img.shape[1]}")

    # 6. Factor pool definitions
    print(f"\n[FACTOR_POOLS]")
    for name, indices in FACTOR_POOLS.items():
        print(f"  {name:15s} → classes {indices}  (n={len(indices)})")
        print(f"    formula: pooled = mean(mean_cams[{indices}], axis=0), then min-max normalised")

    print(f"\n{sep}\n")


# ════════════════════════════════════════════════
# GradCAM-based neuropil importance
# ════════════════════════════════════════════════

NEUROPIL_GROUPS = {
    'Odor':       [0, 1, 2, 3],
    'Taste':      [4, 5, 6, 7],
    'Combined':   [8, 9, 10, 11, 12, 13, 14, 15],
    'Appetitive': [0, 2, 4, 6, 8, 12],
    'Aversive':   [1, 3, 5, 7, 9, 13],
    'Conflict':   [10, 11, 14, 15],
    'Starved':    [0, 1, 4, 5, 8, 9, 10, 11],
    'Fed':        [2, 3, 6, 7, 12, 13, 14, 15],
}

NEUROPIL_CONTRASTS = {
    'Starved_minus_Fed':          ('Starved',    'Fed'),
    'Odor_minus_Taste':           ('Odor',       'Taste'),
    'Appetitive_minus_Aversive':  ('Appetitive', 'Aversive'),
}


def compute_gradcam_per_sample(model, test_loader, target_layer, device, logger=None):
    """Run GradCAM++ on every test sample; return only correctly classified ones.

    Returns
    -------
    correct_cams   : ndarray (N_correct, 128, 128)
    correct_labels : ndarray (N_correct,)
    correct_paths  : list[str]
    """
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    def _log(msg):
        if logger: logger.info(msg)
        else: print(msg)

    model.eval()

    # ── forward pass: collect predictions ─────────────────────────────────
    all_preds_list = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            out = model(X_batch.to(device))
            logits = out[0] if isinstance(out, tuple) else out
            all_preds_list.append(logits.argmax(dim=1).cpu().numpy())

    all_preds  = np.concatenate(all_preds_list)
    all_labels = np.array(test_loader.dataset.labels)
    correct_idx = np.where(all_preds == all_labels)[0]
    N_correct = len(correct_idx)
    _log(f'Correctly classified: {N_correct}/{len(all_labels)} '
         f'({100 * N_correct / len(all_labels):.1f} %)')

    # ── GradCAM++ per sample ───────────────────────────────────────────────
    cam_engine = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    cam_list = []

    for X_batch, y_batch in test_loader:
        B   = X_batch.shape[0]
        y_np = y_batch.numpy()

        if X_batch.dim() == 5:
            _, S, C, H, W = X_batch.shape
            frames  = X_batch.reshape(B * S, C, H, W).to(device)
            targets = [ClassifierOutputTarget(int(y_np[i]))
                       for i in range(B) for _ in range(S)]
        else:
            S       = 1
            frames  = X_batch.to(device)
            targets = [ClassifierOutputTarget(int(y_np[i])) for i in range(B)]

        g_cams = cam_engine(input_tensor=frames, targets=targets)

        if S > 1:
            Hc, Wc = g_cams.shape[-2], g_cams.shape[-1]
            g_cams = g_cams.reshape(B, S, Hc, Wc).mean(axis=1)

        for i in range(B):
            gc = g_cams[i]
            if gc.shape[0] != 128 or gc.shape[1] != 128:
                t  = torch.tensor(gc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                gc = F.interpolate(t, size=(128, 128), mode='bilinear',
                                   align_corners=False).squeeze().numpy()
            cam_list.append(gc)

    del cam_engine

    all_cams       = np.stack(cam_list)
    correct_cams   = all_cams[correct_idx]
    correct_labels = all_labels[correct_idx]
    correct_paths  = [test_loader.dataset.image_paths[i] for i in correct_idx]
    return correct_cams, correct_labels, correct_paths


def load_neuropil_masks(correct_paths, allTs_path, neuropil_names,
                        mask_size=128, logger=None):
    """Load per-sample binary neuropil masks from isolated-neuropil TIFFs.

    Returns
    -------
    masks : bool ndarray (N_correct, n_neuropils, mask_size, mask_size)
    """
    def _log(msg):
        if logger: logger.info(msg)
        else: print(msg)

    N = len(correct_paths)
    n_np = len(neuropil_names)
    masks = np.zeros((N, n_np, mask_size, mask_size), dtype=bool)

    _mask_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((mask_size, mask_size),
                          interpolation=transforms.InterpolationMode.NEAREST),
    ])

    for j, neuropil in enumerate(neuropil_names):
        neuropil_base = f'{allTs_path}_{neuropil}'
        n_missing = 0
        for i, p in enumerate(correct_paths):
            np_path = os.path.join(
                neuropil_base,
                os.path.basename(os.path.dirname(p)),
                os.path.basename(p),
            )
            if not os.path.exists(np_path):
                n_missing += 1
                continue
            raw        = tifffile.imread(np_path)
            img_t      = _mask_tf(raw)
            masks[i, j] = img_t.numpy()[0] > 0
        if n_missing:
            _log(f'{neuropil}: {n_missing}/{N} mask files missing')

    return masks


def compute_gradcam_neuropil_importance_by_group(
    correct_cams, correct_labels, masks, neuropil_names,
    groups=None, contrasts=None,
):
    """Compute mean GradCAM intensity within each neuropil mask, aggregated by group.

    Returns
    -------
    df_group    : DataFrame (n_groups, n_neuropils)
    df_contrast : DataFrame (n_contrasts, n_neuropils)
    """
    if groups is None:
        groups = NEUROPIL_GROUPS
    if contrasts is None:
        contrasts = NEUROPIL_CONTRASTS

    N, n_np = correct_cams.shape[0], len(neuropil_names)
    importance = np.full((N, n_np), np.nan, dtype=np.float32)
    for i in range(N):
        for j in range(n_np):
            m = masks[i, j]
            if m.sum() > 0:
                importance[i, j] = correct_cams[i][m].mean()

    group_profiles = {}
    for gname, cls_idx in groups.items():
        sel = np.isin(correct_labels, cls_idx)
        group_profiles[gname] = np.nanmean(importance[sel], axis=0)

    df_group = pd.DataFrame(group_profiles, index=neuropil_names).T

    contrast_profiles = {}
    for cname, (g1, g2) in contrasts.items():
        contrast_profiles[cname] = group_profiles[g1] - group_profiles[g2]

    df_contrast = pd.DataFrame(contrast_profiles, index=neuropil_names).T
    return df_group, df_contrast


# ════════════════════════════════════════════════
# Plotting: Pooled GradCAM grid (3×3)
# ════════════════════════════════════════════════

def plot_gradcam_pooled(
    fig, gs_slot, pooled_cams,
    sketch_path=None, cmap='viridis',
    normalize_per_row=False,
    normalize_global=True,
    brain_shape=None,
    y_shift=0.0,
):
    """
    Plot pooled GradCAM as a tight 3-column × 3-row grid (transposed).

    Columns = factor groups  (State, Modality, Valence)
    Rows    = individual factors within each group.
    Labels on the LEFT of each image (vertical ylabel).
    Column headers above top row only.
    Colorbar in its own thin column on the far right.

    Layout:
              State       Modality      Valence     ┃cbar
    Row 0:  ┃ Fed       ┃ Odor        ┃ App.       ┃ ▐
    Row 1:  ┃ Starved   ┃ Taste       ┃ Avers.     ┃ ▐
    Row 2:  ┃ (empty)   ┃ Odor+Taste  ┃ App./Avers.┃ ▐
    """
    # Columns: group label → list of factor names (top-to-bottom)
    columns = [
        ('State',    ['Fed',    'Starved',   None]),
        ('Modality', ['Odor',   'Taste',     'Odor+Taste']),
        ('Valence',  ['App.',   'Avers.',    'Mixed']),
    ]

    display_labels = {
        'Fed': 'Fed', 'Starved': 'Starved',
        'Odor': 'Odor', 'Taste': 'Taste', 'Odor+Taste': 'Odor+Taste',
        'App.': 'App.', 'Avers.': 'Avers.', 'Mixed': 'App./Avers.',
    }

    nrows, ncols = 3, 3
    pooled_cams = dict(pooled_cams)          # shallow copy

    # ── Global normalisation ───────────────────────────────
    if normalize_global:
        all_maps = [pooled_cams[e]
                    for _, entries in columns
                    for e in entries if e is not None]
        stack = np.stack(all_maps)
        gmin, gmax = stack.min(), stack.max()
        for _, entries in columns:
            for e in entries:
                if e is None:
                    continue
                if gmax - gmin > 1e-8:
                    pooled_cams[e] = (pooled_cams[e] - gmin) / (gmax - gmin)
                else:
                    pooled_cams[e] = pooled_cams[e] * 0.0

    # ── Per-row normalisation (rows in the transposed sense) ──
    elif normalize_per_row:
        for r in range(nrows):
            valid = [pooled_cams[columns[c][1][r]]
                     for c in range(ncols)
                     if columns[c][1][r] is not None]
            if not valid:
                continue
            row_stack = np.stack(valid)
            rmin, rmax = row_stack.min(), row_stack.max()
            for c in range(ncols):
                e = columns[c][1][r]
                if e is None:
                    continue
                if rmax - rmin > 1e-8:
                    pooled_cams[e] = (pooled_cams[e] - rmin) / (rmax - rmin)
                else:
                    pooled_cams[e] = pooled_cams[e] * 0.0

    # ── Sub-gridspec: 3 rows × 4 cols (3 images + 1 colorbar) ──
    inner_gs = GridSpecFromSubplotSpec(
        nrows, ncols + 1,
        subplot_spec=gs_slot,
        wspace=0.04, hspace=0.05,
        width_ratios=[1, 1, 1, 0.1],
    )

    # ── Image grid ─────────────────────────────────────────
    im_ref = None
    for c, (col_label, entries) in enumerate(columns):
        for r, entry in enumerate(entries):
            ax = fig.add_subplot(inner_gs[r, c])

            if entry is None:
                ax.axis('off')
                continue

            cam = pooled_cams[entry]
            if brain_shape is not None:
                t   = torch.tensor(cam, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                cam = F.interpolate(t, size=brain_shape, mode='bilinear',
                                    align_corners=False).squeeze().numpy()
            im = ax.imshow(cam, cmap=cmap, aspect='equal', vmin=0, vmax=1)
            if im_ref is None:
                im_ref = im

            # Factor label horizontal above each cell
            lbl = display_labels.get(entry, entry)
            ax.set_title(lbl, fontsize=FONT_SIZES['annotation'], pad=3)

            # Column group header (State / Modality / Valence) above the factor label, top row only
            if r == 0:
                ax.text(0.5, 1.30, col_label,
                        transform=ax.transAxes, ha='center', va='bottom',
                        fontsize=FONT_SIZES['annotation'], fontweight='bold',
                        clip_on=False)

            ax.set_xticks([]); ax.set_yticks([])

    # ── Colorbar column (spans all rows) ───────────────────
    if im_ref is not None:

        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(ax)
        # ax_cb = divider.append_axes("right", size="3%", pad=0.08)

        ax_cb = fig.add_subplot(inner_gs[:, ncols])
        fig.colorbar(im_ref, cax=ax_cb)
        ax_cb.set_ylabel('GradCAM intensity', fontsize=FONT_SIZES['colorbar'], labelpad=6)

    # ── Vertical shift (positive = up, negative = down) ────
    if y_shift != 0.0:
        for ax in fig.get_axes():
            pos = ax.get_position()
            # only move axes that live inside gs_slot's bounding box
            slot_bb = gs_slot.get_position(fig)
            if (pos.x0 >= slot_bb.x0 - 0.01 and pos.x1 <= slot_bb.x1 + 0.01 and
                    pos.y0 >= slot_bb.y0 - 0.01 and pos.y1 <= slot_bb.y1 + 0.01):
                ax.set_position([pos.x0, pos.y0 + y_shift, pos.width, pos.height])

    return fig


# ════════════════════════════════════════════════
# Plotting: Heatmaps
# ════════════════════════════════════════════════

def plot_heatmap_groups_abs(
    ax, group_profiles_abs, neuropil_names=None,
    cmap='viridis', annotate=True, fontsize_annot=None, title=None,
    normalize=True,
):
    """Grouped absolute importance heatmap."""
    if neuropil_names is None:
        neuropil_names = list(group_profiles_abs.columns)
    values = group_profiles_abs.values
    if normalize:
        vmin, vmax = values.min(), values.max()
        if vmax - vmin > 1e-8:
            values = (values - vmin) / (vmax - vmin)
    _label_map = {'Multi': 'Combined', 'Valence_Mix': 'Conflict'}
    group_names = [_label_map.get(n, n) for n in group_profiles_abs.index]

    if fontsize_annot is None:
        fontsize_annot = FONT_SIZES['heatmap_cell']
    # vmax=0.7 clips the top so the 0-0.7 range uses the full colormap spread.
    # To revert to linear full range, replace vmax=0.7 with vmax=1.
    im = ax.imshow(values, aspect='auto', cmap=cmap, vmin=0, vmax=0.7)
    ax.set_xticks(range(len(neuropil_names)))
    ax.set_xticklabels(neuropil_names, rotation=45, ha='right', fontsize=FONT_SIZES['tick'])
    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(group_names, fontsize=FONT_SIZES['tick'])
    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i,j]:.2f}",
                        ha='center', va='center', fontsize=fontsize_annot, color='white')
    if title:
        ax.set_title(title)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_label('Importance Weights', fontsize=FONT_SIZES['colorbar'], labelpad=6)
    return ax


# ════════════════════════════════════════════════
# Plotting: Horizontal contrast barplots
# ════════════════════════════════════════════════

DEFAULT_CONTRAST_SPEC = {
    'State': {
        'key': 'Starved_minus_Fed',
        'pos_label': 'Fed', 'neg_label': 'Starved',
        'pos_color': '#404040', 'neg_color': '#808080',
    },
    'Modality': {
        'key': 'Odor_minus_Taste',
        'pos_label': 'Taste', 'neg_label': 'Odor',
        'pos_color': '#3382BE', 'neg_color': '#D35F2A',
    },
    'Valence': {
        'key': 'Appetitive_minus_Aversive',
        'pos_label': 'Aversive', 'neg_label': 'Appetitive',
        'pos_color': '#D62728', 'neg_color': '#2CA02C',
    },
}


def plot_contrasts_horizontal(
    axes, group_contrasts_abs, neuropil_names=None,
    contrast_spec=None, uniform_xlim=True,
    sort_by_modality=False,
    fontsize_title=None, fontsize_labels=None, fontsize_legend=None,
):
    """
    3-panel HORIZONTAL contrast barplots.
    Shared x-axis, no tick numbers, scalebar on bottom-right of last panel.
    Optionally sort neuropils by modality contrast (odor-most → taste-most).
    """
    if contrast_spec is None:
        contrast_spec = DEFAULT_CONTRAST_SPEC
    if neuropil_names is None:
        neuropil_names = list(group_contrasts_abs.columns)
    if fontsize_title is None:
        fontsize_title = FONT_SIZES['subplot_title']
    if fontsize_labels is None:
        fontsize_labels = FONT_SIZES['label']
    if fontsize_legend is None:
        fontsize_legend = FONT_SIZES['legend']

    n_np = len(neuropil_names)
    neuropil_names = list(neuropil_names)  # ensure mutable copy

    # Sort by modality importance: most Odor at top → most Taste at bottom
    if sort_by_modality and 'Odor_minus_Taste' in group_contrasts_abs.index:
        mod_vals = group_contrasts_abs.loc['Odor_minus_Taste'].values
        sort_idx = np.argsort(mod_vals)[::-1]  # descending: Odor-dominant first
        neuropil_names_sorted = [neuropil_names[i] for i in sort_idx]
    else:
        sort_idx = np.arange(n_np)
        neuropil_names_sorted = neuropil_names

    if uniform_xlim:
        all_vals = np.concatenate([
            group_contrasts_abs.loc[info['key']].values
            for info in contrast_spec.values()
            if info['key'] in group_contrasts_abs.index
        ])
        xlim_max = np.abs(all_vals).max() * 1.15
    else:
        xlim_max = None

    for ax_idx, (ax, (title, info)) in enumerate(zip(axes, contrast_spec.items())):
        key = info['key']
        if key not in group_contrasts_abs.index:
            ax.text(0.5, 0.5, f'Missing: {key}', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        vals = -group_contrasts_abs.loc[key].values[sort_idx]  # negate to flip direction
        colors = [info['pos_color'] if v >= 0 else info['neg_color'] for v in vals]

        y_pos = np.arange(n_np)
        ax.barh(y_pos, vals, color=colors, height=0.7)
        ax.set_yticks(y_pos)
        ax.invert_yaxis()
        ax.axvline(0, color='black', linewidth=0.5)

        if xlim_max is not None:
            ax.set_xlim(-xlim_max, xlim_max)

        # Tight y-limits
        ax.set_ylim(n_np - 0.5, -0.5)

        # Title below
        ax.set_xlabel(title, fontsize=fontsize_title, fontweight='bold')

        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        neg_lbl = info["neg_label"].replace("Appetitive", "App.").replace("Aversive", "Avs.")
        pos_lbl = info["pos_label"].replace("Appetitive", "App.").replace("Aversive", "Avs.")


        # Direction annotation above

        ax.text(0, 1.03, '|', ha='center', va='bottom', fontsize=fontsize_labels, transform=trans)
        ax.text(0, 1.03, f'← {neg_lbl} ', ha='right', va='bottom', fontsize=fontsize_labels, transform=trans)
        ax.text(0, 1.03, f' {pos_lbl} →', ha='left', va='bottom', fontsize=fontsize_labels, transform=trans)
        # ax.text(0.5, 1.03,
        #         f'← {info["neg_label"]}  |  {info["pos_label"]} →',
        #         ha='center', va='bottom', fontsize=14,
        #         transform=ax.transAxes)

        # Y-tick labels + left spine only on FIRST panel
        if ax_idx == 0:
            ax.set_yticklabels(neuropil_names_sorted, fontsize=fontsize_labels)
            ax.spines['left'].set_visible(False)
            # Draw y-axis line from first to last neuropil
            xmin = ax.get_xlim()[0]
            ax.plot([xmin, xmin], [-0.5, n_np - 0.5],
                    color='black', linewidth=0.8, clip_on=False)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)
            ax.spines['left'].set_visible(False)

        # Remove x-tick numbers and unnecessary spines
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    # Scalebar on bottom-right corner of last (valence) panel
    if xlim_max is not None:
        scale_val = 0.1
        last_ax = axes[-1]
        sb_x_start = xlim_max * 0.4
        sb_x_end = sb_x_start + scale_val
        sb_y = n_np - 0.5  # just below last neuropil
        last_ax.plot([sb_x_start, sb_x_end], [sb_y, sb_y],
                     color='black', linewidth=1.5,
                     clip_on=False, solid_capstyle='butt')

    return axes


# ════════════════════════════════════════════════
# Plotting: Diverging heatmap (KO ΔAccuracy)
# ════════════════════════════════════════════════

def plot_heatmap_groups_diverging(
    ax, group_profiles, neuropil_names=None,
    cmap='RdBu_r', annotate=True, fontsize_annot=None,
    title=None, cbar_label='Mean ΔAccuracy [pp]',
):
    """Group-level heatmap with a diverging colourmap centred at 0.

    Designed for KO ΔAccuracy values (positive = neuropil matters for the group).
    """
    if neuropil_names is None:
        neuropil_names = list(group_profiles.columns)
    values     = group_profiles.values
    group_names = list(group_profiles.index)

    if fontsize_annot is None:
        fontsize_annot = FONT_SIZES['heatmap_cell']

    vmax = np.nanpercentile(np.abs(values), 95)
    im = ax.imshow(values, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(neuropil_names)))
    ax.set_xticklabels(neuropil_names, rotation=45, ha='right',
                       fontsize=FONT_SIZES['tick'])
    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(group_names, fontsize=FONT_SIZES['tick'])

    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                text_color = 'white' if abs(v) > 0.5 * vmax else 'black'
                ax.text(j, i, f'{v:.2f}',
                        ha='center', va='center',
                        fontsize=fontsize_annot, color=text_color)

    if title:
        ax.set_title(title, fontsize=FONT_SIZES['subplot_title'])

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax  = divider.append_axes('right', size='3%', pad=0.08)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, fontsize=FONT_SIZES['colorbar'], labelpad=6)
    return ax


# ════════════════════════════════════════════════
# Plotting: Correlation matrix
# ════════════════════════════════════════════════

def plot_correlation_matrix(
    ax, corr_df, annotate=True, fontsize_annot=None,
    title="Inter-neuropil correlation",
):
    """Neuropil correlation matrix."""
    if fontsize_annot is None:
        fontsize_annot = FONT_SIZES['heatmap_cell']
    names = list(corr_df.columns)
    values = corr_df.values
    im = ax.imshow(values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i,j]:.2f}",
                        ha='center', va='center', fontsize=fontsize_annot)
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)
    return ax