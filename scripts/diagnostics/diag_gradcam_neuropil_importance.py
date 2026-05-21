"""
Diagnostic: GradCAM++ × neuropil-mask overlap importance
=========================================================
For each correctly classified test sample, derives the 12 recording-specific
2D neuropil masks from the pre-computed isolated-neuropil TIFF images
(binarised non-zero pixels = 2D projection of the 3D volumetric mask) and
averages the per-sample GradCAM++ intensity within each mask.

Aggregates by condition group, produces panel-e/f equivalents, and prints
Spearman rank correlations against the existing ridge |W| ranking.

Outputs (results/diagnostics/gradcam_neuropil/):
    panel_e_equivalent.pdf      — group importance heatmap (8 groups × 12 neuropils)
    panel_f_equivalent.pdf      — contrast bar charts (3 panels)
    group_profiles_gradcam.csv  — raw group means
    group_contrasts_gradcam.csv — raw contrast profiles

Everything is self-contained in this file.  Delete it and the repo is untouched.
"""

import os
import pickle

import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import tifffile
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import CustomDataset
from src.models.cnn_transformer import CNN_Transformer
from src.models.model_io import load_model
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.visualization.figure_base import apply_style, FONT_SIZES

apply_style()

# ── constants ──────────────────────────────────────────────────────────────

OUT_DIR = 'results/diagnostics/gradcam_neuropil'
os.makedirs(OUT_DIR, exist_ok=True)

GRADCAM_LAYER = 'conv1'
LAYER_MAP = {'conv1': 0, 'conv2': 3, 'conv3': 6}

# Order matches Table T3
NEUROPILS = ['AL', 'MB', 'PENP', 'VLNP', 'CX', 'GNG',
             'LX', 'SNP', 'INP', 'LH', 'OL', 'VMNP']

GROUPS = {
    'Odor':       [0, 1, 2, 3],
    'Taste':      [4, 5, 6, 7],
    'Combined':   [8, 9, 10, 11, 12, 13, 14, 15],
    'Appetitive': [0, 2, 4, 6, 8, 12],
    'Aversive':   [1, 3, 5, 7, 9, 13],
    'Conflict':   [10, 11, 14, 15],
    'Starved':    [0, 1, 4, 5, 8, 9, 10, 11],
    'Fed':        [2, 3, 6, 7, 12, 13, 14, 15],
}
GROUP_ORDER = list(GROUPS.keys())

CONTRASTS = {
    'Odor_minus_Taste':          ('Odor',       'Taste'),
    'Appetitive_minus_Aversive': ('Appetitive', 'Aversive'),
    'Starved_minus_Fed':         ('Starved',    'Fed'),
}

# Visual convention matches the existing interpretability figure (panel f)
CONTRAST_SPEC = {
    'State': {
        'key':       'Starved_minus_Fed',
        'neg_label': 'Starved', 'pos_label': 'Fed',
        'neg_color': '#808080', 'pos_color': '#404040',
    },
    'Modality': {
        'key':       'Odor_minus_Taste',
        'neg_label': 'Odor',     'pos_label': 'Taste',
        'neg_color': '#D35F2A',  'pos_color': '#3382BE',
    },
    'Valence': {
        'key':       'Appetitive_minus_Aversive',
        'neg_label': 'Appetitive', 'pos_label': 'Aversive',
        'neg_color': '#2CA02C',    'pos_color': '#D62728',
    },
}

# Ridge reference data produced by scripts/analysis/run_neuropil_importance.py
RIDGE_DATADIR = '/rhomes/aabdel/DrosoEmbedding/results/Neuropil_Importance/aktuell/data'

RANK_FLAG_THRESHOLD = 4   # flag neuropils whose rank shifts by this many positions

# ── config ────────────────────────────────────────────────────────────────

config = load_config()
logger = setup_logger(task_name=config['run_id'],
                      log_dir='logs/diag_gradcam_neuropil')

# Preserve yaml-derived paths (pickle_path, models dir) before overwrite
paths        = config['paths']
model_params = config['model']['parameters']
train_params = config['training']

with open(f"{paths['results_root']}/config.pkl", 'rb') as fh:
    config = pickle.load(fh)

device     = config['device']
allTs_path = config['paths']['allTs_path']   # e.g. .../meanZ_allTs

# ── test data ─────────────────────────────────────────────────────────────

with open(paths['pickle_path'], 'rb') as fh:
    _, _, X_test, _, _, Y_test = pickle.load(fh)

test_dataset = CustomDataset(
    X_test, Y_test, transform=True,
    seq_length=model_params['seq_len'],
    seq_steps=model_params['seq_steps'],
    allTs_path=allTs_path,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=train_params['batch_size'],
    shuffle=False, num_workers=4, pin_memory=True,
)
N = len(test_dataset)
logger.info(f'Test set: {N} samples after sequence filtering')

# ── model ─────────────────────────────────────────────────────────────────

classifier, _, _, _, _, _ = load_model(
    CNN_Transformer, model_params, paths['models'], device, logger
)
if classifier is None:
    raise FileNotFoundError('Trained model not found.')
classifier.eval()

# ── ridge reference ───────────────────────────────────────────────────────

W_abs          = np.load(os.path.join(RIDGE_DATADIR, 'W_abs.npy'))       # (12, 16)
ridge_group_df = pd.read_csv(
    os.path.join(RIDGE_DATADIR, 'group_profiles_abs.csv'), index_col=0
)   # rows = group names, columns = neuropil names

# ── STEP 1: forward pass — identify correctly classified samples ───────────

logger.info('STEP 1: collecting predictions')

all_preds_list = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        out      = classifier(X_batch.to(device))
        logits_b = out[0] if isinstance(out, tuple) else out
        all_preds_list.append(logits_b.argmax(dim=1).cpu().numpy())

all_preds   = np.concatenate(all_preds_list)
all_labels  = np.array(test_dataset.labels)
correct_idx = np.where(all_preds == all_labels)[0]
N_correct   = len(correct_idx)
logger.info(f'Correctly classified: {N_correct}/{N}  ({100 * N_correct / N:.1f} %)')

# ── STEP 2: per-sample GradCAM++ ──────────────────────────────────────────
# Iterate once in dataset order; target = each sample's true class.
# Sequence frames are averaged into a single (128, 128) map per sample.

logger.info('STEP 2: computing per-sample GradCAM++')

target_layer = classifier.cnn[LAYER_MAP[GRADCAM_LAYER]]
cam_engine   = GradCAMPlusPlus(model=classifier, target_layers=[target_layer])
cam_list     = []   # one (128, 128) array per sample, in test_dataset order

for X_batch, y_batch in test_loader:
    B    = X_batch.shape[0]
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

    g_cams = cam_engine(input_tensor=frames, targets=targets)   # (B*S, H', W')

    if S > 1:
        H_c, W_c = g_cams.shape[-2], g_cams.shape[-1]
        g_cams = g_cams.reshape(B, S, H_c, W_c).mean(axis=1)   # (B, H', W')

    for i in range(B):
        gc = g_cams[i]
        if gc.shape[0] != 128 or gc.shape[1] != 128:
            t  = torch.tensor(gc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            gc = F.interpolate(t, size=(128, 128), mode='bilinear',
                               align_corners=False).squeeze().numpy()
        cam_list.append(gc)

del cam_engine

all_cams       = np.stack(cam_list)               # (N, 128, 128)
correct_cams   = all_cams[correct_idx]            # (N_correct, 128, 128)
correct_labels = all_labels[correct_idx]          # (N_correct,)
correct_paths  = [test_dataset.image_paths[i] for i in correct_idx]

# ── STEP 3: recording-specific neuropil masks ──────────────────────────────
# For each neuropil, the mask is the binarised non-zero region of the
# pre-computed isolated-neuropil TIFF (2D projection of the 3D volumetric mask).
# Nearest-neighbour resize to 128×128 avoids introducing non-zero boundary artefacts.

logger.info('STEP 3: loading neuropil masks')

_mask_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128),
                      interpolation=transforms.InterpolationMode.NEAREST),
])

masks = np.zeros((N_correct, 12, 128, 128), dtype=bool)

for j, neuropil in enumerate(NEUROPILS):
    # String-swap: append _{neuropil} to the base allTs path to get the
    # isolated-neuropil directory — equivalent to what setup_derived_parameters
    # produces with isolate_neuropil=True, without mutating any shared config.
    neuropil_base = f'{allTs_path}_{neuropil}'
    n_missing = 0

    for i, p in enumerate(correct_paths):
        np_path = os.path.join(
            neuropil_base,
            os.path.basename(os.path.dirname(p)),   # RecordingID
            os.path.basename(p),                    # RecordingID_frame.tiff
        )
        if not os.path.exists(np_path):
            n_missing += 1
            continue
        raw         = tifffile.imread(np_path)
        img_t       = _mask_tf(raw)                 # (1, 128, 128)
        masks[i, j] = img_t.numpy()[0] > 0

    if n_missing:
        logger.warning(f'{neuropil}: {n_missing}/{N_correct} mask files missing — '
                       f'affected samples contribute NaN for this neuropil')

# ── STEP 4: GradCAM × mask importance ─────────────────────────────────────

logger.info('STEP 4: computing importance vectors')

importance = np.full((N_correct, 12), np.nan, dtype=np.float32)
for i in range(N_correct):
    for j in range(12):
        m = masks[i, j]
        if m.sum() > 0:
            importance[i, j] = correct_cams[i][m].mean()

# ── STEP 5: group aggregation ─────────────────────────────────────────────

logger.info('STEP 5: aggregating by condition group')

group_profiles = {}
for gname, cls_idx in GROUPS.items():
    sel                   = np.isin(correct_labels, cls_idx)
    group_profiles[gname] = np.nanmean(importance[sel], axis=0)   # (12,)

df_group = pd.DataFrame(group_profiles, index=NEUROPILS).T        # (8, 12)
df_group.to_csv(os.path.join(OUT_DIR, 'group_profiles_gradcam.csv'))

contrast_profiles = {}
for cname, (g1, g2) in CONTRASTS.items():
    contrast_profiles[cname] = group_profiles[g1] - group_profiles[g2]

df_contrast = pd.DataFrame(contrast_profiles, index=NEUROPILS).T
df_contrast.to_csv(os.path.join(OUT_DIR, 'group_contrasts_gradcam.csv'))

# ── STEP 6: Spearman rank summary ─────────────────────────────────────────

logger.info('STEP 6: Spearman rank correlations vs. ridge |W|')

SEP = '=' * 64
print(f'\n{SEP}')
print('GradCAM++ × mask  vs.  ridge |W|  —  Spearman rank correlations')
print(SEP)

for gname in GROUP_ORDER:
    if gname not in ridge_group_df.index:
        print(f'  {gname:<14}  [no matching ridge row — skipped]')
        continue
    gc_vec  = df_group.loc[gname, NEUROPILS].values.astype(float)
    rdg_vec = ridge_group_df.loc[gname, NEUROPILS].values.astype(float)
    if np.isnan(gc_vec).any():
        print(f'  {gname:<14}  [NaN in GradCAM values — skipped]')
        continue

    rho, pval = spearmanr(gc_vec, rdg_vec)
    print(f'  {gname:<14}  ρ = {rho:+.3f}  (p = {pval:.3f})')

    # Rank 0 = highest importance; flag neuropils that shift substantially
    gc_ranks  = np.argsort(np.argsort(gc_vec[::-1]))
    rdg_ranks = np.argsort(np.argsort(rdg_vec[::-1]))
    deltas    = np.abs(gc_ranks - rdg_ranks)
    for k in np.where(deltas >= RANK_FLAG_THRESHOLD)[0]:
        print(f'    *** {NEUROPILS[k]}: GradCAM rank {gc_ranks[k] + 1}'
              f'  vs. ridge rank {rdg_ranks[k] + 1}  (Δ = {deltas[k]})')

# Overall Spearman on the full flattened (8 × 12) matrix
shared   = [g for g in GROUP_ORDER
            if g in ridge_group_df.index and g in df_group.index]
gc_flat  = np.array([df_group.loc[g, NEUROPILS].values for g in shared],
                    dtype=float).ravel()
rdg_flat = np.array([ridge_group_df.loc[g, NEUROPILS].values for g in shared],
                    dtype=float).ravel()
valid    = ~np.isnan(gc_flat)
rho_all, pval_all = spearmanr(gc_flat[valid], rdg_flat[valid])
print(f'\n  Overall ({len(shared)} groups × 12, flattened):'
      f'  ρ = {rho_all:+.3f}  (p = {pval_all:.4f})')
print(f'{SEP}\n')

# ── STEP 7: figures ───────────────────────────────────────────────────────

logger.info('STEP 7: generating figures')

# ── panel e equivalent: group importance heatmap ──────────────────────────

vals_e = np.array([df_group.loc[g, NEUROPILS].values for g in GROUP_ORDER])

fig_e, ax_e = plt.subplots(figsize=(10, 5))
im = ax_e.imshow(vals_e, aspect='auto', cmap='viridis')
ax_e.set_xticks(range(12))
ax_e.set_xticklabels(NEUROPILS, rotation=45, ha='right',
                     fontsize=FONT_SIZES['tick'])
ax_e.set_yticks(range(len(GROUP_ORDER)))
ax_e.set_yticklabels(GROUP_ORDER, fontsize=FONT_SIZES['tick'])
for i in range(vals_e.shape[0]):
    for j in range(vals_e.shape[1]):
        ax_e.text(j, i, f'{vals_e[i, j]:.3f}',
                  ha='center', va='center',
                  fontsize=FONT_SIZES['heatmap_cell'], color='white')
divider = make_axes_locatable(ax_e)
cax_e   = divider.append_axes('right', size='3%', pad=0.08)
cbar_e  = fig_e.colorbar(im, cax=cax_e)
cbar_e.set_label('Mean GradCAM++ intensity within mask',
                 fontsize=FONT_SIZES['colorbar'], labelpad=6)
ax_e.set_title(
    f'GradCAM++ neuropil importance by condition group'
    f'  (N = {N_correct} correctly classified)',
    fontsize=FONT_SIZES['subplot_title'],
)
path_e = os.path.join(OUT_DIR, 'panel_e_equivalent.pdf')
fig_e.savefig(path_e, bbox_inches='tight')
fig_e.savefig(path_e.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
plt.close(fig_e)
logger.info(f'Saved: {path_e}')

# ── panel f equivalent: contrast bar charts ───────────────────────────────
# Neuropils sorted by Odor_minus_Taste, matching the existing figure convention.

sort_idx_f      = np.argsort(contrast_profiles['Odor_minus_Taste'])[::-1]
neuropil_sorted = [NEUROPILS[k] for k in sort_idx_f]
n_np            = 12
xlim_max        = max(
    np.abs(v[sort_idx_f]).max() for v in contrast_profiles.values()
) * 1.15

fig_f, axes_f = plt.subplots(1, 3, figsize=(12, 5))

for ax_idx, (ax, (panel_key, info)) in enumerate(zip(axes_f, CONTRAST_SPEC.items())):
    # Negate to match the sign convention used in the existing figure
    vals_f  = -contrast_profiles[info['key']][sort_idx_f]
    colours = [info['pos_color'] if v >= 0 else info['neg_color'] for v in vals_f]
    y_pos   = np.arange(n_np)

    ax.barh(y_pos, vals_f, color=colours, height=0.7)
    ax.set_yticks(y_pos)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlim(-xlim_max, xlim_max)
    ax.set_ylim(n_np - 0.5, -0.5)
    ax.set_xlabel(panel_key, fontsize=FONT_SIZES['subplot_title'], fontweight='bold')

    neg_lbl = info['neg_label'].replace('Appetitive', 'App.').replace('Aversive', 'Avs.')
    pos_lbl = info['pos_label'].replace('Appetitive', 'App.').replace('Aversive', 'Avs.')
    trans   = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(0, 1.03, '|',
            ha='center', va='bottom', fontsize=FONT_SIZES['label'], transform=trans)
    ax.text(0, 1.03, f'← {neg_lbl} ',
            ha='right',  va='bottom', fontsize=FONT_SIZES['label'], transform=trans)
    ax.text(0, 1.03, f' {pos_lbl} →',
            ha='left',   va='bottom', fontsize=FONT_SIZES['label'], transform=trans)

    if ax_idx == 0:
        ax.set_yticklabels(neuropil_sorted, fontsize=FONT_SIZES['label'])
        ax.spines['left'].set_visible(False)
        xmin = ax.get_xlim()[0]
        ax.plot([xmin, xmin], [-0.5, n_np - 0.5],
                color='black', linewidth=0.8, clip_on=False)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)

plt.tight_layout()
path_f = os.path.join(OUT_DIR, 'panel_f_equivalent.pdf')
fig_f.savefig(path_f, bbox_inches='tight')
fig_f.savefig(path_f.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
plt.close(fig_f)
logger.info(f'Saved: {path_f}')

logger.info('Done.')
