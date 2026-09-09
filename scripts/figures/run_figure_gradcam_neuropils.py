"""
Figure: Neuropil Importance (GradCAM-based)
============================================
2 × 3 layout — all panels derived from GradCAM attribution maps.

Top row (width_ratios = [0.6, 1.1, 0.8]):
┌──────────────┬──────────────────┬──────────────┐
│ a) GradCAM   │ b) GradCAM       │ c) Neuropil  │
│    pipeline  │    attribution   │    atlas     │
│    sketch    │    maps (3×3)    │    sketch    │
└──────────────┴──────────────────┴──────────────┘
Attribution maps are upscaled to BRAIN_SHAPE derived from the mean raw
frame dimensions (height fixed at 128, width scaled by the measured aspect ratio).

Bottom row (2 equal columns, independent GridSpec):
┌──────────────────────┬──────────────────────────┐
│ d) Neuropil          │ e) Contrast plots        │
│    importance        │    State | Mod. | Valence│
│    heatmap (GradCAM) │    (GradCAM-based)       │
└──────────────────────┴──────────────────────────┘
"""

import os
import pickle

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from src.models.model_io import load_model
from src.models.cnn_transformer import CNN_Transformer
from src.data.dataset import CustomDataset
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from torch.utils.data import DataLoader

from src.visualization.visualize_interpretability import (
    compute_gradcam,
    pool_gradcams,
    compute_gradcam_per_sample,
    load_neuropil_masks,
    compute_gradcam_neuropil_importance_by_group,
    plot_gradcam_pooled,
    plot_heatmap_groups_abs,
    plot_contrasts_horizontal,
    load_image,
)
from src.data.dataset import compute_mean_frame_shape
from src.visualization.figure_base import (
    apply_style, FIGURE_WIDTH, save_figure, add_panel_label,
)

apply_style()

# ════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════

GRADCAM_LAYER = 'conv1'
LAYER_MAP = {'conv1': 0, 'conv2': 3, 'conv3': 6}

NEUROPILS = ['AL', 'MB', 'PENP', 'VLNP', 'CX', 'GNG',
             'LX', 'SNP', 'INP', 'LH', 'OL', 'VMNP']

# ════════════════════════════════════════════════
# LOAD CONFIG + MODEL + DATA
# ════════════════════════════════════════════════

config = load_config()
logger = setup_logger(task_name=config['run_id'],
                      log_dir='logs/run_figure_neuropils')

paths        = config['paths']
OUT_DIR             = paths['output_dir']
OUT_STEM            = f'fig_neuropils_{GRADCAM_LAYER}'
GRADCAM_SKETCH_PATH = os.path.join(paths['src_imgs_dir'], 'CAM_Sketch.svg')
ATLAS_PATH          = os.path.join(paths['src_imgs_dir'], 'NeuropilsAtlas.svg')
model_params = config['model']['parameters']
train_params = config['training']

with open(f"{paths['results_root']}/config.pkl", 'rb') as fh:
    config = pickle.load(fh)

device     = config['device']
allTs_path = config['paths']['allTs_path']

with open(paths['pickle_path'], 'rb') as fh:
    _, _, X_test, _, _, Y_test = pickle.load(fh)

test_dataset = CustomDataset(
    X_test, Y_test, transform=True,
    seq_length=model_params['seq_len'],
    seq_steps=model_params['seq_steps'],
    allTs_path=allTs_path,
)
test_loader = DataLoader(
    test_dataset, batch_size=train_params['batch_size'],
    shuffle=False, num_workers=4, pin_memory=True,
)

classifier, _, _, _, _, _ = load_model(
    CNN_Transformer, model_params, paths['models'], device, logger,
)
if classifier is None:
    raise FileNotFoundError('Trained model not found.')

class_names = config['data']['classes']
n_classes   = len(class_names)

# ════════════════════════════════════════════════
# COMPUTE GRADCAM (pooled, for panel b)
# ════════════════════════════════════════════════

logger.info(f'Computing pooled GradCAM [{GRADCAM_LAYER}] for panel b…')
target_layer = classifier.cnn[LAYER_MAP[GRADCAM_LAYER]]

mean_cams, _ = compute_gradcam(
    model=classifier, test_loader=test_loader,
    target_layer=target_layer, n_classes=n_classes,
    device=device, logger=logger,
)
pooled_cams = pool_gradcams(mean_cams)

# ════════════════════════════════════════════════
# COMPUTE GRADCAM NEUROPIL IMPORTANCE (panels d, e)
# ════════════════════════════════════════════════

logger.info('Computing per-sample GradCAM for neuropil importance…')
correct_cams, correct_labels, correct_paths = compute_gradcam_per_sample(
    model=classifier, test_loader=test_loader,
    target_layer=target_layer, device=device, logger=logger,
)

logger.info('Computing mean raw frame shape for GradCAM upscaling…')
mean_H, mean_W = compute_mean_frame_shape(correct_paths, allTs_path, n_sample=200, logger=logger)
BRAIN_SHAPE = (128, round(128 * mean_W / mean_H))
logger.info(f'BRAIN_SHAPE set to {BRAIN_SHAPE}')

logger.info('Loading neuropil masks…')
masks = load_neuropil_masks(
    correct_paths=correct_paths,
    allTs_path=allTs_path,
    neuropil_names=NEUROPILS,
    mask_size=128,
    logger=logger,
)

logger.info('Aggregating GradCAM importance by group…')
df_group, df_contrast = compute_gradcam_neuropil_importance_by_group(
    correct_cams=correct_cams,
    correct_labels=correct_labels,
    masks=masks,
    neuropil_names=NEUROPILS,
)

# ════════════════════════════════════════════════
# ASSEMBLE FIGURE — 2 × 3
# ════════════════════════════════════════════════
#
#  Top row  (3 columns, width_ratios [0.6, 1.1, 0.8]):
#  ┌──────────────┬──────────────────┬──────────────┐
#  │ a) GradCAM   │ b) attribution   │ c) atlas     │
#  │    sketch    │    maps 3×3      │              │
#  └──────────────┴──────────────────┴──────────────┘
#
#  Bottom row (2 equal columns, independent GridSpec):
#  ┌──────────────────────┬──────────────────────────┐
#  │ d) heatmap           │ e) contrasts             │
#  └──────────────────────┴──────────────────────────┘

logger.info('Assembling figure…')

fig = plt.figure(figsize=(FIGURE_WIDTH, 13))

gs_top = GridSpec(
    1, 3, figure=fig,
    left=0.01, right=0.97,
    bottom=0.52, top=0.93,
    wspace=0.25,
    width_ratios=[0.6, 1.1, 0.8],
)

gs_bot = GridSpec(
    1, 2, figure=fig,
    left=0.05, right=0.97,
    bottom=0.05, top=0.47,
    wspace=0.30,
)

# ── Panel labels ──────────────────────────────────────────────────────────
add_panel_label(fig, 'a', x=0.00, y=0.96)
add_panel_label(fig, 'b', x=0.25, y=0.96)
add_panel_label(fig, 'c', x=0.69, y=0.96)
add_panel_label(fig, 'd', x=0.00, y=0.50)
add_panel_label(fig, 'e', x=0.52, y=0.50)

# ═══════════════════ TOP ROW ═══════════════════

# ── Panel a: GradCAM pipeline sketch ──────────────────────────────────────
ax_sketch = fig.add_subplot(gs_top[0, 0])
if os.path.exists(GRADCAM_SKETCH_PATH):
    sketch_img = load_image(GRADCAM_SKETCH_PATH)
    ax_sketch.imshow(sketch_img, aspect='equal')
ax_sketch.set_xticks([]); ax_sketch.set_yticks([])
for spine in ax_sketch.spines.values():
    spine.set_visible(False)

# ── Panel b: GradCAM attribution maps 3×3 — upscaled to BRAIN_SHAPE ──────
plot_gradcam_pooled(
    fig=fig, gs_slot=gs_top[0, 1],
    pooled_cams=pooled_cams,
    normalize_global=True,
    normalize_per_row=False,
    brain_shape=BRAIN_SHAPE,
    y_shift=-0.025,             # shift panel b down so group headers align with 'b' label
    hspace=-0.085,              # negative = rows overlap slightly, tightens vertical extent
    cbar_y=(0.516, 0.882),      # colorbar y extent in figure coordinates
)

# ── Panel c: Neuropil atlas sketch ────────────────────────────────────────
ax_atlas = fig.add_subplot(gs_top[0, 2])
if os.path.exists(ATLAS_PATH):
    atlas_img = load_image(ATLAS_PATH)
    ax_atlas.imshow(atlas_img, aspect='equal')
    pos = ax_atlas.get_position()
    ax_atlas.set_position([pos.x0 - 0.01, pos.y0, pos.width, pos.height])
ax_atlas.set_xticks([]); ax_atlas.set_yticks([])
for spine in ax_atlas.spines.values():
    spine.set_visible(False)

# ═══════════════════ BOTTOM ROW ═══════════════════

# ── Panel d: GradCAM neuropil importance heatmap ──────────────────────────
ax_heat = fig.add_subplot(gs_bot[0, 0])
plot_heatmap_groups_abs(
    ax=ax_heat,
    group_profiles_abs=df_group,
    neuropil_names=NEUROPILS,
    annotate=False,
    cbar_label='Mean Grad-CAM++ attribution',
)

# ── Panel e: GradCAM contrast plots (3 sub-panels) ────────────────────────
gs_contrasts = GridSpecFromSubplotSpec(
    1, 3, subplot_spec=gs_bot[0, 1], wspace=0.06,
)
axes_contrast = [fig.add_subplot(gs_contrasts[0, i]) for i in range(3)]
plot_contrasts_horizontal(
    axes=axes_contrast,
    group_contrasts_abs=df_contrast,
    neuropil_names=NEUROPILS,
    sort_by_modality=True,
    show_scalebar=False,
)

# ════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════

os.makedirs(OUT_DIR, exist_ok=True)
save_figure(fig, os.path.join(OUT_DIR, f'{OUT_STEM}.pdf'),
            formats=('png', 'svg', 'pdf'))
logger.info('Done.')
