"""
Diagnostic: neuropil importance — three-method comparison
==========================================================
1. Mask overlap quantification  — pairwise IoU between 2D-projected masks
2. Mask-aggregated GradCAM++    — per-condition-group heatmap + contrast bars
3. Shuffle test                 — ΔAccuracy when mask pixels are permuted
4. Three-way Spearman + biological flags

Mask loading replicates neuropils_importance_cofficients.py STEP 2 exactly:
full 5-frame neuropil sequence loaded via CustomDataset, frames averaged,
binarised > 0. Label order verified with the same assert.

Outputs (results/diagnostics/neuropil_methods/):
    mask_overlap.pdf/png
    panel_e_equivalent.pdf/png
    panel_f_equivalent.pdf/png
    shuffle_importance.pdf/png
    group_profiles_gradcam.csv / group_contrasts_gradcam.csv

Delete this file to leave the repository untouched.
"""

import os
import pickle

import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from src.data.dataset import CustomDataset
from src.models.cnn_transformer import CNN_Transformer
from src.models.model_io import load_model
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.visualization.figure_base import apply_style, FONT_SIZES

apply_style()

# ── constants ──────────────────────────────────────────────────────────────

OUT_DIR = 'results/diagnostics/neuropil_methods'
os.makedirs(OUT_DIR, exist_ok=True)

GRADCAM_LAYER = 'conv1'
LAYER_MAP     = {'conv1': 0, 'conv2': 3, 'conv3': 6}

# Table T3 order
NEUROPILS = ['AL', 'MB', 'PENP', 'VLNP', 'CX', 'GNG',
             'LX', 'SNP', 'INP', 'LH', 'OL', 'VMNP']
N_NP = len(NEUROPILS)

# British spelling throughout; ridge CSV uses American names — see RIDGE_GROUP_RENAME
GROUPS = {
    'Odour':      [0, 1, 2, 3],
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
    'Odour_minus_Taste':         ('Odour', 'Taste'),
    'Appetitive_minus_Aversive': ('Appetitive', 'Aversive'),
    'Starved_minus_Fed':         ('Starved', 'Fed'),
}

CONTRAST_SPEC = {
    'State': {
        'key': 'Starved_minus_Fed',
        'neg_label': 'Starved', 'pos_label': 'Fed',
        'neg_color': '#808080',  'pos_color': '#404040',
    },
    'Modality': {
        'key': 'Odour_minus_Taste',
        'neg_label': 'Odour',   'pos_label': 'Taste',
        'neg_color': '#D35F2A', 'pos_color': '#3382BE',
    },
    'Valence': {
        'key': 'Appetitive_minus_Aversive',
        'neg_label': 'Appetitive', 'pos_label': 'Aversive',
        'neg_color': '#2CA02C',    'pos_color': '#D62728',
    },
}

# Rename ridge CSV group index to match British spelling above
RIDGE_GROUP_RENAME = {'Odor': 'Odour', 'Conflicting': 'Conflict'}

RIDGE_DATADIR = '/rhomes/aabdel/DrosoEmbedding/results/Neuropil_Importance/aktuell/data'

RNG_BASE_SEED       = 42
K_SHUFFLES          = 5
RANK_FLAG_THRESHOLD = 4    # flag rank shifts ≥ this
IOU_FLAG_THRESHOLD  = 0.3  # flag pairs with mean IoU above this

# Known anatomical priors: (neuropil, check_type, args, rationale)
BIOL_PRIORS = [
    ('AL',  'gradcam_group_gt', ('Odour', 'Taste'),
     'Antennal lobe — primary olfactory relay — should favour Odour'),
    ('LH',  'gradcam_group_gt', ('Odour', 'Taste'),
     'Lateral horn — downstream olfactory processing — should favour Odour'),
    ('GNG', 'gradcam_group_gt', ('Taste', 'Odour'),
     'Gnathal ganglion (suboesophageal zone) — gustatory relay — should favour Taste'),
    ('MB',  'gradcam_valence_diff', None,
     'Mushroom body — encodes reward vs. punishment — expect Appetitive ≠ Aversive'),
    ('OL',  'shuffle_rank_low', 10,
     'Optic lobe — visual processing only — no visual stimuli, expect low shuffle ΔAcc'),
]

# ── helpers ────────────────────────────────────────────────────────────────

def save_fig(fig, stem):
    """Save as PDF and PNG (300 dpi)."""
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT_DIR, f'{stem}.{ext}'),
                    dpi=300 if ext == 'png' else None,
                    bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {stem}.pdf / .png')


def descending_ranks(vec):
    """Return 1-based ranks, rank 1 = highest value."""
    return np.argsort(np.argsort(vec)[::-1]) + 1


# ── config ─────────────────────────────────────────────────────────────────

config = load_config()
logger = setup_logger(task_name=config['run_id'],
                      log_dir='logs/diag_neuropil_methods_comparison')

paths        = config['paths']
model_params = config['model']['parameters']
train_params = config['training']

with open(f"{paths['results_root']}/config.pkl", 'rb') as fh:
    config = pickle.load(fh)

device     = config['device']
allTs_path = config['paths']['allTs_path']   # e.g. .../meanZ_allTs

# ── test data (main full-brain images) ────────────────────────────────────

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
N          = len(test_dataset)
Y_filtered = np.array(test_dataset.labels)
logger.info(f'Test set: {N} samples after sequence filtering')

# ── model ──────────────────────────────────────────────────────────────────

classifier, _, _, _, _, _ = load_model(
    CNN_Transformer, model_params, paths['models'], device, logger
)
if classifier is None:
    raise FileNotFoundError('Trained model not found.')
classifier.eval()

# ── ridge reference ────────────────────────────────────────────────────────

W_abs          = np.load(os.path.join(RIDGE_DATADIR, 'W_abs.npy'))        # (12, 16)
ridge_group_df = pd.read_csv(
    os.path.join(RIDGE_DATADIR, 'group_profiles_abs.csv'), index_col=0
).rename(index=RIDGE_GROUP_RENAME)   # normalise to British spelling

# ════════════════════════════════════════════════════════════════════════════
# STEP 1: Load neuropil masks  →  masks (N, 12, 128, 128) bool
# Replicates neuropils_importance_cofficients.py STEP 2 exactly:
#   full 5-frame sequence loaded via CustomDataset, frames averaged, binarised.
# ════════════════════════════════════════════════════════════════════════════

logger.info('STEP 1: loading neuropil masks')

masks = np.zeros((N, N_NP, 128, 128), dtype=bool)

for j, neuropil in enumerate(NEUROPILS):
    # String-swap: append _{neuropil} — equivalent to setup_derived_parameters
    # with isolate_neuropil=True, without mutating any shared config.
    neuropil_base = f'{allTs_path}_{neuropil}'

    X_np = [
        os.path.join(neuropil_base,
                     os.path.basename(os.path.dirname(p)),
                     os.path.basename(p))
        for p in test_dataset.image_paths
    ]

    # Fail fast on missing files (same guard as ridge script)
    for p in X_np[:5]:
        if not os.path.exists(p):
            raise FileNotFoundError(f'{neuropil}: missing {p}')

    ds_np = CustomDataset(
        X_np, Y_filtered.tolist(), transform=True,
        seq_length=model_params['seq_len'],
        seq_steps=model_params['seq_steps'],
        allTs_path=neuropil_base,
    )
    assert len(ds_np) == N, (
        f'{neuropil}: neuropil dataset has {len(ds_np)} samples, '
        f'expected {N}. Check that all neuropil TIFFs exist.'
    )

    loader_np = DataLoader(
        ds_np, batch_size=train_params['batch_size'],
        shuffle=False, num_workers=4, pin_memory=True,
    )

    y_seen = []
    offset = 0
    for X_batch, y_batch in loader_np:
        B = X_batch.shape[0]
        # Average sequence frames — same as ridge script line 170
        if X_batch.dim() == 5:
            neuropil_img = X_batch.mean(dim=1).squeeze(1)    # (B, H, W)
        elif X_batch.dim() == 4:
            neuropil_img = X_batch.squeeze(1)
        else:
            neuropil_img = X_batch

        masks[offset:offset + B, j] = (neuropil_img.cpu().numpy() > 0)
        y_seen.extend(y_batch.numpy().tolist())
        offset += B

    # Same label-order assertion as ridge script
    assert np.array_equal(np.array(y_seen), Y_filtered), \
        f'{neuropil}: label order mismatch with main test set'

    mean_cov = masks[:, j].mean()
    logger.info(f'  {neuropil}: mean coverage = {mean_cov:.4f}')

logger.info('Masks loaded.')

# ════════════════════════════════════════════════════════════════════════════
# STEP 2: Mask overlap quantification
# ════════════════════════════════════════════════════════════════════════════

logger.info('STEP 2: computing pairwise mask overlap')

mean_coverage = np.array([masks[:, j].mean() for j in range(N_NP)])

# Per-sample pairwise IoU, accumulated in batches to stay within memory budget
iou_sum  = np.zeros((N_NP, N_NP))
BATCH_OV = 128

for start in range(0, N, BATCH_OV):
    end  = min(start + BATCH_OV, N)
    m    = masks[start:end].reshape(end - start, N_NP, -1).astype(np.float32)   # (B, 12, HW)
    inter = np.matmul(m, m.transpose(0, 2, 1))                                  # (B, 12, 12)
    sizes = m.sum(axis=2)                                                        # (B, 12)
    union = sizes[:, :, None] + sizes[:, None, :] - inter                       # (B, 12, 12)
    iou   = np.where(union > 0, inter / union, 0.0)
    iou_sum += iou.sum(axis=0)

mean_iou = iou_sum / N   # (12, 12)

# Print summary
SEP = '=' * 64
print(f'\n{SEP}')
print('Mask overlap summary')
print(SEP)
print(f'  Mean coverage per neuropil (fraction of 128×128 pixels):')
for j, (name, cov) in enumerate(zip(NEUROPILS, mean_coverage)):
    print(f'    {name:6s}: {cov:.4f}')

# Off-diagonal pairs sorted by IoU
pairs = [(mean_iou[i, j], NEUROPILS[i], NEUROPILS[j])
         for i in range(N_NP) for j in range(i + 1, N_NP)]
pairs.sort(reverse=True)
print(f'\n  Top-10 overlapping pairs (mean IoU):')
flagged_pairs = []
for iou_val, a, b in pairs[:10]:
    flag = '  *** SUBSTANTIAL' if iou_val > IOU_FLAG_THRESHOLD else ''
    print(f'    {a}–{b}: {iou_val:.3f}{flag}')
    if iou_val > IOU_FLAG_THRESHOLD:
        flagged_pairs.append((a, b, iou_val))
if flagged_pairs:
    print(f'\n  Pairs with mean IoU > {IOU_FLAG_THRESHOLD} (potential confounds):')
    for a, b, v in flagged_pairs:
        print(f'    {a}–{b}: {v:.3f}')
print(SEP + '\n')

# ════════════════════════════════════════════════════════════════════════════
# STEP 3: Baseline forward pass
# ════════════════════════════════════════════════════════════════════════════

logger.info('STEP 3: baseline forward pass')

all_preds_list = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        out      = classifier(X_batch.to(device))
        logits_b = out[0] if isinstance(out, tuple) else out
        all_preds_list.append(logits_b.argmax(dim=1).cpu().numpy())

all_preds    = np.concatenate(all_preds_list)
correct_mask = (all_preds == Y_filtered)
correct_idx  = np.where(correct_mask)[0]
N_correct    = len(correct_idx)
baseline_acc = N_correct / N
logger.info(f'Baseline accuracy: {baseline_acc:.4f}  ({N_correct}/{N})')

# ════════════════════════════════════════════════════════════════════════════
# STEP 4: Mask-aggregated GradCAM++  (correctly classified samples only)
# ════════════════════════════════════════════════════════════════════════════

logger.info('STEP 4: computing per-sample GradCAM++')

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
        S, frames = 1, X_batch.to(device)
        targets   = [ClassifierOutputTarget(int(y_np[i])) for i in range(B)]

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
torch.cuda.empty_cache()

all_cams       = np.stack(cam_list)                # (N, 128, 128)
correct_cams   = all_cams[correct_idx]             # (N_correct, 128, 128)
correct_labels = Y_filtered[correct_idx]           # (N_correct,)

# GradCAM importance: mean CAM intensity within each neuropil mask
importance = np.full((N_correct, N_NP), np.nan, dtype=np.float32)
for i in range(N_correct):
    for j in range(N_NP):
        m = masks[correct_idx[i], j]
        if m.sum() > 0:
            importance[i, j] = correct_cams[i][m].mean()

# Group aggregation
gradcam_group_profiles = {}
for gname, cls_idx in GROUPS.items():
    sel = np.isin(correct_labels, cls_idx)
    gradcam_group_profiles[gname] = np.nanmean(importance[sel], axis=0)

df_gradcam_groups = pd.DataFrame(gradcam_group_profiles, index=NEUROPILS).T
df_gradcam_groups.to_csv(os.path.join(OUT_DIR, 'group_profiles_gradcam.csv'))

gradcam_contrast_profiles = {}
for cname, (g1, g2) in CONTRASTS.items():
    gradcam_contrast_profiles[cname] = (gradcam_group_profiles[g1]
                                        - gradcam_group_profiles[g2])

df_gradcam_contrasts = pd.DataFrame(gradcam_contrast_profiles, index=NEUROPILS).T
df_gradcam_contrasts.to_csv(os.path.join(OUT_DIR, 'group_contrasts_gradcam.csv'))

logger.info('GradCAM++ importance computed.')

# ════════════════════════════════════════════════════════════════════════════
# STEP 5: Shuffle test  (all N samples, fixed seed)
# For each neuropil: K shuffles of mask-region pixels in the model input;
# ΔAccuracy = baseline_acc − mean shuffle accuracy.
# ════════════════════════════════════════════════════════════════════════════

logger.info('STEP 5: shuffle test  (12 neuropils × 5 shuffles = 60 forward passes)')

shuffle_accs = np.zeros(N_NP)

for j, neuropil in enumerate(NEUROPILS):
    total_correct = 0
    total_samples = 0

    for k in range(K_SHUFFLES):
        rng           = np.random.default_rng(seed=RNG_BASE_SEED + j * K_SHUFFLES + k)
        sample_offset = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                B = X_batch.shape[0]

                # Build binary mask indices for each sample in this batch
                batch_masks = masks[sample_offset:sample_offset + B, j]  # (B, 128, 128)

                # Shuffle neuropil-j pixels in the full-brain model input.
                # Operate on numpy view of the cloned batch to avoid extra copies.
                X_shuf    = X_batch.clone()   # CPU tensor
                shuf_np   = X_shuf.numpy()    # shares memory: (B, S, C, H, W) or (B, C, H, W)

                if shuf_np.ndim == 5:
                    B_, S_, C_, H_, W_ = shuf_np.shape
                    flat = shuf_np.reshape(B_, S_ * C_, H_ * W_)   # view
                else:
                    B_, C_, H_, W_ = shuf_np.shape
                    S_ = 1
                    flat = shuf_np.reshape(B_, C_, H_ * W_)

                for i in range(B_):
                    pix = np.where(batch_masks[i].ravel())[0]
                    if pix.size == 0:
                        continue
                    # Each row (frame×channel) gets an independent permutation
                    flat[i, :, pix] = rng.permuted(flat[i, :, pix], axis=1)
                # flat is a view of shuf_np which shares memory with X_shuf

                out      = classifier(X_shuf.to(device))
                logits_b = out[0] if isinstance(out, tuple) else out
                preds    = logits_b.argmax(dim=1).cpu().numpy()
                total_correct += (preds == y_batch.numpy()).sum()
                total_samples += B
                sample_offset += B

    shuffle_accs[j] = total_correct / total_samples
    delta = baseline_acc - shuffle_accs[j]
    logger.info(f'  {neuropil}: shuffle_acc = {shuffle_accs[j]:.4f}  '
                f'ΔAcc = {delta:+.4f}')

delta_acc = baseline_acc - shuffle_accs   # (12,)  positive = model relies on this region

logger.info('Shuffle test complete.')

# ════════════════════════════════════════════════════════════════════════════
# STEP 6: Three-way Spearman comparison + biological flags
# ════════════════════════════════════════════════════════════════════════════

logger.info('STEP 6: three-way comparison and biological flags')

# Align each method to a single 12-d overall importance vector
shared_groups = [g for g in GROUP_ORDER if g in ridge_group_df.index]
if len(shared_groups) < len(GROUP_ORDER):
    logger.warning(f'Ridge CSV missing groups: '
                   f'{set(GROUP_ORDER) - set(ridge_group_df.index)}')

ridge_overall  = ridge_group_df.loc[shared_groups, NEUROPILS].values.mean(axis=0)
gradcam_overall = np.array([gradcam_group_profiles[g]
                             for g in shared_groups]).mean(axis=0)
# delta_acc is already a 12-d vector

rho_rg, p_rg = spearmanr(ridge_overall,   gradcam_overall)
rho_rs, p_rs = spearmanr(ridge_overall,   delta_acc)
rho_gs, p_gs = spearmanr(gradcam_overall, delta_acc)

print(f'\n{SEP}')
print('Three-way Spearman rank correlations (overall importance, 12 neuropils)')
print(SEP)
print(f'  Ridge |W|  vs. GradCAM++  : ρ = {rho_rg:+.3f}  (p = {p_rg:.3f})')
print(f'  Ridge |W|  vs. Shuffle ΔAcc: ρ = {rho_rs:+.3f}  (p = {p_rs:.3f})')
print(f'  GradCAM++  vs. Shuffle ΔAcc: ρ = {rho_gs:+.3f}  (p = {p_gs:.3f})')

print(f'\nPer-group GradCAM++ vs. ridge |W|  (Spearman over 12 neuropils):')
for gname in GROUP_ORDER:
    if gname not in ridge_group_df.index:
        print(f'  {gname:<14}  [missing from ridge CSV]')
        continue
    gc_vec  = df_gradcam_groups.loc[gname, NEUROPILS].values.astype(float)
    rdg_vec = ridge_group_df.loc[gname, NEUROPILS].values.astype(float)
    if np.isnan(gc_vec).any():
        print(f'  {gname:<14}  [NaN in GradCAM — skipped]')
        continue
    rho, pval = spearmanr(gc_vec, rdg_vec)
    print(f'  {gname:<14}  ρ = {rho:+.3f}  (p = {pval:.3f})')

    gc_ranks  = descending_ranks(gc_vec)
    rdg_ranks = descending_ranks(rdg_vec)
    deltas    = np.abs(gc_ranks - rdg_ranks)
    for k in np.where(deltas >= RANK_FLAG_THRESHOLD)[0]:
        print(f'    *** {NEUROPILS[k]}: GradCAM rank {gc_ranks[k]}'
              f'  vs. ridge rank {rdg_ranks[k]}  (Δ = {deltas[k]})')

# Neuropils with large rank shifts across all three methods
print(f'\nNeuropil rank comparison across all three methods:')
r_ridge   = descending_ranks(ridge_overall)
r_gradcam = descending_ranks(gradcam_overall)
r_shuffle = descending_ranks(delta_acc)
print(f'  {"Neuropil":8s}  {"Ridge":>6s}  {"GradCAM":>8s}  {"Shuffle":>8s}  '
      f'{"Max Δ":>6s}')
for k, name in enumerate(NEUROPILS):
    max_delta = max(abs(r_ridge[k] - r_gradcam[k]),
                    abs(r_ridge[k] - r_shuffle[k]),
                    abs(r_gradcam[k] - r_shuffle[k]))
    flag = '  ***' if max_delta >= RANK_FLAG_THRESHOLD else ''
    print(f'  {name:8s}  {r_ridge[k]:6d}  {r_gradcam[k]:8d}  '
          f'{r_shuffle[k]:8d}  {max_delta:6d}{flag}')

# Biological flags
print(f'\n{"─" * 64}')
print('Biological interpretation flags')
print(f'{"─" * 64}')
np_idx = {name: i for i, name in enumerate(NEUROPILS)}

for neuropil, check_type, args, rationale in BIOL_PRIORS:
    if neuropil not in np_idx:
        print(f'  [?] {neuropil} not in NEUROPILS list')
        continue

    if check_type == 'gradcam_group_gt':
        g1, g2 = args
        v1 = (df_gradcam_groups.loc[g1, neuropil]
              if g1 in df_gradcam_groups.index else np.nan)
        v2 = (df_gradcam_groups.loc[g2, neuropil]
              if g2 in df_gradcam_groups.index else np.nan)
        if np.isnan(v1) or np.isnan(v2):
            status, detail = '[?]', 'NaN data'
        else:
            passed = v1 > v2
            status = '[BIOL ✓]' if passed else '[BIOL ✗ UNEXPECTED]'
            detail = f'{g1} {v1:.4f} vs {g2} {v2:.4f}'

    elif check_type == 'gradcam_valence_diff':
        app = (df_gradcam_groups.loc['Appetitive', neuropil]
               if 'Appetitive' in df_gradcam_groups.index else np.nan)
        avs = (df_gradcam_groups.loc['Aversive', neuropil]
               if 'Aversive' in df_gradcam_groups.index else np.nan)
        # Threshold: the dataset-wide standard deviation of all group×neuropil values
        threshold = float(df_gradcam_groups.values.std())
        if np.isnan(app) or np.isnan(avs):
            status, detail = '[?]', 'NaN data'
        else:
            diff   = abs(app - avs)
            passed = diff > threshold
            status = '[BIOL ✓]' if passed else '[BIOL ✗ UNEXPECTED]'
            detail = (f'|App ({app:.4f}) − Avers ({avs:.4f})| = {diff:.4f}, '
                      f'threshold = {threshold:.4f}')

    elif check_type == 'shuffle_rank_low':
        threshold = args
        rank   = int(r_shuffle[np_idx[neuropil]])
        passed = rank >= threshold
        status = '[BIOL ✓]' if passed else '[BIOL ✗ UNEXPECTED]'
        detail = f'shuffle ΔAcc rank = {rank}  (expected ≥ {threshold})'

    else:
        status, detail = '[?]', f'unknown check type: {check_type}'

    print(f'  {status} {neuropil}: {detail}')
    print(f'         {rationale}')

print(f'{"─" * 64}\n')

# ════════════════════════════════════════════════════════════════════════════
# STEP 7: Figures
# ════════════════════════════════════════════════════════════════════════════

logger.info('STEP 7: generating figures')

# ── mask overlap heatmap ───────────────────────────────────────────────────

fig_ov, ax_ov = plt.subplots(figsize=(8, 7))
im_ov = ax_ov.imshow(mean_iou, vmin=0, vmax=1, cmap='YlOrRd')
ax_ov.set_xticks(range(N_NP))
ax_ov.set_yticks(range(N_NP))
ax_ov.set_xticklabels(NEUROPILS, rotation=45, ha='right', fontsize=FONT_SIZES['tick'])
ax_ov.set_yticklabels(NEUROPILS, fontsize=FONT_SIZES['tick'])
for i in range(N_NP):
    for j in range(N_NP):
        val = mean_iou[i, j]
        ax_ov.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=FONT_SIZES['heatmap_cell'],
                   color='white' if val > 0.5 else 'black')
div_ov  = make_axes_locatable(ax_ov)
cax_ov  = div_ov.append_axes('right', size='3%', pad=0.08)
cbar_ov = fig_ov.colorbar(im_ov, cax=cax_ov)
cbar_ov.set_label('Mean pairwise IoU', fontsize=FONT_SIZES['colorbar'], labelpad=6)
ax_ov.set_title('Neuropil mask overlap (mean IoU across test recordings)',
                fontsize=FONT_SIZES['subplot_title'])
save_fig(fig_ov, 'mask_overlap')

# ── panel e: GradCAM group importance heatmap ──────────────────────────────

vals_e = np.array([df_gradcam_groups.loc[g, NEUROPILS].values for g in GROUP_ORDER])

fig_e, ax_e = plt.subplots(figsize=(10, 5))
im_e = ax_e.imshow(vals_e, aspect='auto', cmap='viridis')
ax_e.set_xticks(range(N_NP))
ax_e.set_xticklabels(NEUROPILS, rotation=45, ha='right', fontsize=FONT_SIZES['tick'])
ax_e.set_yticks(range(len(GROUP_ORDER)))
ax_e.set_yticklabels(GROUP_ORDER, fontsize=FONT_SIZES['tick'])
for i in range(vals_e.shape[0]):
    for j in range(vals_e.shape[1]):
        ax_e.text(j, i, f'{vals_e[i, j]:.3f}', ha='center', va='center',
                  fontsize=FONT_SIZES['heatmap_cell'], color='white')
div_e  = make_axes_locatable(ax_e)
cax_e  = div_e.append_axes('right', size='3%', pad=0.08)
cbar_e = fig_e.colorbar(im_e, cax=cax_e)
cbar_e.set_label('Mean GradCAM++ intensity within mask',
                 fontsize=FONT_SIZES['colorbar'], labelpad=6)
ax_e.set_title(
    f'GradCAM++ neuropil importance by condition group'
    f'  (N = {N_correct} correctly classified)',
    fontsize=FONT_SIZES['subplot_title'],
)
save_fig(fig_e, 'panel_e_equivalent')

# ── panel f: GradCAM contrast bars ────────────────────────────────────────

sort_idx_f     = np.argsort(gradcam_contrast_profiles['Odour_minus_Taste'])[::-1]
neuropil_sorted = [NEUROPILS[k] for k in sort_idx_f]
xlim_max_f      = max(
    np.abs(v[sort_idx_f]).max() for v in gradcam_contrast_profiles.values()
) * 1.15

fig_f, axes_f = plt.subplots(1, 3, figsize=(12, 5))

for ax_idx, (ax, (panel_key, info)) in enumerate(zip(axes_f, CONTRAST_SPEC.items())):
    vals_f  = -gradcam_contrast_profiles[info['key']][sort_idx_f]
    colours = [info['pos_color'] if v >= 0 else info['neg_color'] for v in vals_f]
    y_pos   = np.arange(N_NP)

    ax.barh(y_pos, vals_f, color=colours, height=0.7)
    ax.set_yticks(y_pos)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlim(-xlim_max_f, xlim_max_f)
    ax.set_ylim(N_NP - 0.5, -0.5)
    ax.set_xlabel(panel_key, fontsize=FONT_SIZES['subplot_title'], fontweight='bold')

    neg_lbl = info['neg_label'].replace('Appetitive', 'App.').replace('Aversive', 'Avs.')
    pos_lbl = info['pos_label'].replace('Appetitive', 'App.').replace('Aversive', 'Avs.')
    trans   = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(0, 1.03, '|', ha='center', va='bottom',
            fontsize=FONT_SIZES['label'], transform=trans)
    ax.text(0, 1.03, f'← {neg_lbl} ', ha='right', va='bottom',
            fontsize=FONT_SIZES['label'], transform=trans)
    ax.text(0, 1.03, f' {pos_lbl} →', ha='left', va='bottom',
            fontsize=FONT_SIZES['label'], transform=trans)

    if ax_idx == 0:
        ax.set_yticklabels(neuropil_sorted, fontsize=FONT_SIZES['label'])
        ax.spines['left'].set_visible(False)
        xmin = ax.get_xlim()[0]
        ax.plot([xmin, xmin], [-0.5, N_NP - 0.5],
                color='black', linewidth=0.8, clip_on=False)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)

plt.tight_layout()
save_fig(fig_f, 'panel_f_equivalent')

# ── shuffle importance bar chart ───────────────────────────────────────────

sort_idx_s      = np.argsort(delta_acc)[::-1]
neuropils_shuf  = [NEUROPILS[k] for k in sort_idx_s]
delta_sorted    = delta_acc[sort_idx_s]
colours_shuf    = ['#2CA02C' if v > 0 else '#D62728' for v in delta_sorted]

fig_s, ax_s = plt.subplots(figsize=(6, 7))
y_pos_s = np.arange(N_NP)
ax_s.barh(y_pos_s, delta_sorted, color=colours_shuf, height=0.7)
ax_s.set_yticks(y_pos_s)
ax_s.set_yticklabels(neuropils_shuf, fontsize=FONT_SIZES['tick'])
ax_s.invert_yaxis()
ax_s.axvline(0, color='black', linewidth=0.8)
ax_s.set_xlabel('ΔAccuracy  (baseline − shuffled)',
                fontsize=FONT_SIZES['label'])
ax_s.set_title(
    f'Shuffle test: neuropil importance\n'
    f'K = {K_SHUFFLES} shuffles per neuropil  ·  seed = {RNG_BASE_SEED}',
    fontsize=FONT_SIZES['subplot_title'],
)
# Annotate bars with values
for i, v in enumerate(delta_sorted):
    pad = 0.0005
    ha  = 'left' if v >= 0 else 'right'
    ax_s.text(v + pad if v >= 0 else v - pad, i, f'{v:+.3f}',
              va='center', ha=ha, fontsize=FONT_SIZES['small'])
plt.tight_layout()
save_fig(fig_s, 'shuffle_importance')

logger.info('Done.')
