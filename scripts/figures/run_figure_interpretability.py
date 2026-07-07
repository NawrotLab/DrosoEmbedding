"""
Figure 4: Model Interpretability  (v2 — new layout)
=====================================================
Pooled GradCAM (8 factors), neuropil importance heatmap, horizontal contrasts,
pipeline sketch, and neuropil atlas.

NEW layout (2 × 3, independent row widths):
    Top row  (width_ratios = [0.8, 1.6, 1.2]):
    ┌──────────────┬──────────────────┬──────────────┐
    │ a) GradCAM   │ b) GradCAM       │ c) Anatomy   │
    │    pipeline  │    attribution   │              │
    │    sketch    │    map (3×3)     │    (atlas)   │
    └──────────────┴──────────────────┴──────────────┘

    Bottom row (width_ratios = [0.6, 1.2, 1.2]):
    ┌──────────┬───────────────────┬──────────────────┐
    │ d) Ridge  │ e) Neuropil       │ f) Diff.         │
    │    weight │    importance     │    importance    │
    │    map    │    (heatmap) □    │    (bars)        │
    └──────────┴───────────────────┴──────────────────┘
    e is square and the same width as f.

Changes vs. original
---------------------
- GradCAM normalised **per row** (State / Modality / Valence independently)
- Saves 20 random GradCAM (input, attribution) example pairs
- Prints sizes of every data element
- Saves PNG + SVG + PDF vector outputs
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import cairosvg
from PIL import Image
import io

from src.models.model_io import load_model
from src.models.cnn_transformer import CNN_Transformer
from src.data.dataset import CustomDataset
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from torch.utils.data import DataLoader


from src.visualization.visualize_interpretability import (
    compute_gradcam,
    pool_gradcams,
    load_neuropil_data,
    plot_gradcam_pooled,
    plot_heatmap_groups_abs,
    plot_contrasts_horizontal,
    save_gradcam_examples,
    save_feature_maps,
    print_element_sizes,
    load_image,
    add_panel_label
)
from src.visualization.figure_base import apply_style, FONT_SIZES, FIGURE_WIDTH, save_figure

apply_style()

# ════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════

GRADCAM_LAYER = "conv1"

config = load_config()
logger = setup_logger(task_name=config['run_id'], log_dir="logs/run_figure_interpretability")

config_path = f"{config['paths']['results_root']}/config.pkl"
paths = config['paths']
model_params = config['model']['parameters']
training_params = config['training']

with open(config_path, 'rb') as f:
    config = pickle.load(f)

DATADIR = "/rhomes/aabdel/DrosoEmbedding/results/Neuropil_Importance/aktuell/data"
OUT_DIR = "/rhomes/aabdel/DrosoEmbedding/results/CombiPlots"
OUT_STEM = f"fig_Interpretability_{GRADCAM_LAYER}"
EXAMPLES_DIR = os.path.join(OUT_DIR, "gradcam_examples")
FEATUREMAPS_DIR = os.path.join(OUT_DIR, "feature_maps")

LAYER_MAP = {
    'conv1': 0,   # cnn[0] = first Conv2d  → 64×64
    'conv2': 3,   # cnn[3] = second Conv2d  → 16×16
    'conv3': 6,   # cnn[6] = third Conv2d   → 4×4
}

# Image paths
SKETCH_PATH = "/rhomes/aabdel/DrosoEmbedding/src/src_imgs/RidgeSketch.svg"
GRADCAM_SKETCH_PATH = "/rhomes/aabdel/DrosoEmbedding/src/src_imgs/CAM_Sketch.svg"
ATLAS_PATH = "/rhomes/aabdel/DrosoEmbedding/src/src_imgs/NeuropilsAtlas.svg"

# ════════════════════════════════════════════════
# LOAD MODEL + DATA
# ════════════════════════════════════════════════

logger.info("Loading model and test data...")

with open(paths["pickle_path"], 'rb') as f:
    _, _, X_test, _, _, Y_test = pickle.load(f)

test_dataset = CustomDataset(
    X_test, Y_test, transform=True,
    seq_length=model_params['seq_len'],
    seq_steps=model_params['seq_steps'],
    allTs_path=config['paths']['allTs_path'],
)
test_loader = DataLoader(
    test_dataset, batch_size=training_params['batch_size'],
    shuffle=False, num_workers=4, pin_memory=True,
)

classifier, _, _, _, _, _ = load_model(
    CNN_Transformer, model_params, paths['models'], config['device'], logger
)
if classifier is None:
    raise FileNotFoundError("Model not found.")

class_names = config['data']['classes']
n_classes = len(class_names)

np_data = load_neuropil_data(DATADIR)
neuropil_names = np_data['neuropil_names']
logger.info(f"Loaded: {n_classes} classes, {len(neuropil_names)} neuropils")

# ════════════════════════════════════════════════
# COMPUTE GRADCAM + POOL
# ════════════════════════════════════════════════

logger.info(f"Computing GradCAM for layer: {GRADCAM_LAYER} (cnn[{LAYER_MAP[GRADCAM_LAYER]}])")
target_layer = classifier.cnn[LAYER_MAP[GRADCAM_LAYER]]

# ── NEW: compute_gradcam now also returns individual examples ──
mean_cams, all_examples = compute_gradcam(
    model=classifier, test_loader=test_loader,
    target_layer=target_layer, n_classes=n_classes,
    device=config['device'], logger=logger,
)

pooled_cams = pool_gradcams(mean_cams)
logger.info(f"Pooled GradCAMs: {list(pooled_cams.keys())}")

# ════════════════════════════════════════════════
# SAVE 20 RANDOM GRADCAM EXAMPLES
# ════════════════════════════════════════════════

logger.info("Saving 20 random GradCAM examples...")
selected_examples = save_gradcam_examples(
    all_examples=all_examples,
    out_dir=EXAMPLES_DIR,
    n=20,
    class_names=class_names,
    cmap='viridis',
    seed=42,
    logger=logger,
)

# ════════════════════════════════════════════════
# SAVE FEATURE MAPS FROM TARGET LAYER (same 20 samples)
# ════════════════════════════════════════════════

logger.info(f"Saving feature maps from layer: {GRADCAM_LAYER} (same samples)...")
save_feature_maps(
    model=classifier,
    target_layer=target_layer,
    selected_examples=selected_examples,
    out_dir=FEATUREMAPS_DIR,
    class_names=class_names,
    cmap='viridis',
    device=config['device'],
    logger=logger,
)

# ════════════════════════════════════════════════
# PRINT ELEMENT SIZES
# ════════════════════════════════════════════════

print_element_sizes(
    mean_cams=mean_cams,
    pooled_cams=pooled_cams,
    np_data=np_data,
    class_names=class_names,
    sketch_path=SKETCH_PATH,
    atlas_path=ATLAS_PATH,
    gradcam_sketch_path=GRADCAM_SKETCH_PATH,
)

# ════════════════════════════════════════════════
# ASSEMBLE FIGURE — NEW 2×3 LAYOUT
# ════════════════════════════════════════════════
#
#  Top row  (width_ratios = [0.8, 1.6, 1.2]):
#  ┌──────────────┬──────────────────┬──────────────┐
#  │ a) GradCAM   │ b) GradCAM       │ c) Anatomy   │
#  │    pipeline  │    attribution   │              │
#  │    sketch    │    map (3×3)     │    (atlas)   │
#  └──────────────┴──────────────────┴──────────────┘
#
#  Bottom row (width_ratios = [0.6, 1.2, 1.2]):
#  ┌──────────┬───────────────────┬──────────────────┐
#  │ d) Ridge  │ e) Neuropil       │ f) Diff.         │
#  │    weight │    importance     │    importance    │
#  │    map    │    (heatmap) □    │    (bars)        │
#  └──────────┴───────────────────┴──────────────────┘
#
#  e is square and the same width as f.
#

logger.info("Assembling figure (new 2×3 layout, independent row widths)...")

fig = plt.figure(figsize=(FIGURE_WIDTH, 13))

# ── Two independent GridSpecs — one per row ────────────────
#    They share left/right/hspace margins but have their own
#    width_ratios so the bottom row can give e and f equal room.

gs_top = GridSpec(
    1, 3, figure=fig,
    left=0.05, right=0.97,
    bottom=0.52, top=0.93,
    wspace=0.30,
    width_ratios=[0.6, 1.1, 0.8],
)

gs_bot = GridSpec(
    1, 3, figure=fig,
    left=0.05, right=0.97,
    bottom=0.05, top=0.47,
    wspace=0.30,
    width_ratios=[0.6, 1.1, 0.8],
)


fig.text(0.07, 0.925, 'a', ha='left', fontsize=FONT_SIZES['panel_label'], fontweight='bold')
fig.text(0.25, 0.925, 'b', ha='left', fontsize=FONT_SIZES['panel_label'], fontweight='bold')
fig.text(0.69, 0.925, 'c', ha='left', fontsize=FONT_SIZES['panel_label'], fontweight='bold')
fig.text(0.07, 0.50, 'd', ha='left', fontsize=FONT_SIZES['panel_label'], fontweight='bold')
fig.text(0.25, 0.50, 'e', ha='left', fontsize=FONT_SIZES['panel_label'], fontweight='bold')
fig.text(0.69, 0.50, 'f', ha='left', fontsize=FONT_SIZES['panel_label'], fontweight='bold')

# ═══════════════════ TOP ROW ═══════════════════

# ── Panel a: GradCAM pipeline sketch (top-left) ────────────
ax_pipeline = fig.add_subplot(gs_top[0, 0])
if os.path.exists(GRADCAM_SKETCH_PATH):
    sketch_img = load_image(GRADCAM_SKETCH_PATH)
    ax_pipeline.imshow(sketch_img, aspect='equal')
    ax_pipeline_pos = ax_pipeline.get_position()
    ax_pipeline.set_position([ax_pipeline_pos.x0 + 0.02, ax_pipeline_pos.y0, ax_pipeline_pos.width, ax_pipeline_pos.height])

ax_pipeline.set_xticks([]); ax_pipeline.set_yticks([])
for spine in ax_pipeline.spines.values():
    spine.set_visible(False)

# add_panel_label(fig, ax_pipeline, 'a')


# ── Panel b: GradCAM attribution maps 3×3 (top-middle) ────
gs_gradcam = gs_top[0, 1]
plot_gradcam_pooled(
    fig=fig, gs_slot=gs_gradcam,
    pooled_cams=pooled_cams,
    sketch_path=None,
    normalize_global=True,      # ← all maps share one colour range
    normalize_per_row=False,
    y_shift=-0.025,             # shift panel b down so group headers align with 'b' label
    hspace=-0.07,               # negative = rows overlap slightly, tightens vertical extent
    cbar_y=(0.55, 0.85),        # colorbar y extent in figure coordinates
)

# ax_gradcam_pos = gs_gradcam.get_position()
# gs_gradcam.set_position([ax_gradcam_pos.x0 - 0.01, ax_gradcam_pos.y0, ax_gradcam_pos.width, ax_gradcam_pos.height])
# ── Panel c: Anatomy / Neuropil atlas (top-right) ─────────
ax_atlas = fig.add_subplot(gs_top[0, 2])
# add_panel_label(fig, ax_atlas, 'c')

if os.path.exists(ATLAS_PATH):
    # atlas_img = mpimg.imread(ATLAS_PATH)
    atlas_img = load_image(ATLAS_PATH)
    ax_atlas.imshow(atlas_img, aspect='equal')
    ax_atlas_pos = ax_atlas.get_position()
    ax_atlas.set_position([ax_atlas_pos.x0 - 0.01, ax_atlas_pos.y0, ax_atlas_pos.width, ax_atlas_pos.height])
ax_atlas.set_xticks([]); ax_atlas.set_yticks([])
for spine in ax_atlas.spines.values():
    spine.set_visible(False)

# ═══════════════════ BOTTOM ROW ═══════════════════

# ── Panel d: Ridge weight map sketch (bottom-left, narrow) ──
ax_ridge = fig.add_subplot(gs_bot[0, 0])
# add_panel_label(fig, ax_ridge, 'd')

if os.path.exists(SKETCH_PATH):
    # ridge_img = mpimg.imread(SKETCH_PATH)
    ridge_img = load_image(SKETCH_PATH)
    ax_ridge.imshow(ridge_img, aspect='equal')
ax_ridge.set_xticks([]); ax_ridge.set_yticks([])
for spine in ax_ridge.spines.values():
    spine.set_visible(False)

# ── Panel e: Neuropil importance heatmap (bottom-middle, SQUARE) ──
ax_heat = fig.add_subplot(gs_bot[0, 1])
# add_panel_label(fig, ax_heat, 'e')
plot_heatmap_groups_abs(
    ax=ax_heat,
    group_profiles_abs=np_data['group_profiles_abs'],
    neuropil_names=neuropil_names,
)
# ax_heat.set_aspect('equal', adjustable='box')       # ← force square

# ── Panel f: Differential importance bars (bottom-right) ──
gs_contrasts = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_bot[0, 2], wspace=0.06)
axes_contrast = [fig.add_subplot(gs_contrasts[0, i]) for i in range(3)]
plot_contrasts_horizontal(
    axes=axes_contrast,
    group_contrasts_abs=np_data['group_contrasts_abs'],
    neuropil_names=neuropil_names,
    sort_by_modality=True,
)

# ════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════

save_figure(fig, os.path.join(OUT_DIR, f"{OUT_STEM}.pdf"), formats=('png', 'svg', 'eps', 'pdf'))
logger.info("Done.")