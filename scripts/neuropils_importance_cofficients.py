"""
Neuropil Feature Importance Analysis (v2 — with permutation test)
==================================================================
Fits a ridge regression from regional mean activity (12 neuropils) to model logits (16 classes).
Ridge weights quantify which neuropils' input activity levels are most predictive of the model's output.

Changes vs. original
---------------------
- Added permutation test (1000 permutations) for R² significance
- Results saved to new directory to avoid overwriting original data

Outputs:
  Figures (in OUTDIR):
    - heatmap_classes_abs.png       : |W| heatmap, 16 classes x 12 neuropils
    - heatmap_groups_abs.png        : |W| heatmap, grouped conditions x 12 neuropils
    - fig4_contrasts_3panel.png     : dW contrast barplots (Modality, Valence, State)
    - summary_barplots_factors.png  : Aggregated |W| per factor (Modality, State, Valence)
    - summary_barplot_overall.png   : Overall |W| sorted
    - correlation_matrix.png        : Inter-neuropil correlation matrix

  Data (in OUTDIR/data/):
    - W_abs.npy / W_abs.csv         : Absolute ridge weights (12x16)
    - W_signed.npy / W_signed.csv   : Signed ridge weights (12x16)
    - neuropil_correlation.csv      : Pairwise neuropil correlations
    - group_profiles_abs.csv        : Grouped absolute profiles
    - group_contrasts_abs.csv       : Contrast profiles (absolute)
    - neuropil_names.json           : Neuropil labels
    - ridge_summary.json            : Alpha, R2 (train + CV), per-class R2, permutation test
"""

from src.models.model_io import load_model
from src.utils.config_loader import load_config, setup_derived_parameters
from torch.utils.data import DataLoader
from src.data.dataset import CustomDataset
from src.models.cnn_transformer import CNN_Transformer
from src.utils.logger import setup_logger

from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import pickle, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from src.visualization.figure_base import apply_style, FONT_SIZES, PAGE_WIDTH

apply_style()

# ================================================================
# CONFIG
# ================================================================

config = load_config()
logger = setup_logger(task_name=config['run_id'], log_dir="logs/neuropils_importance_cofficients_v2")

NEUROPILS = ['AL', 'MB', 'PENP', 'VLNP', 'CX', 'GNG', 'LX', 'SNP', 'INP', 'LH', 'OL', 'VMNP']

run_id = config['run_id']
config_path = f"{config['paths']['results_root']}/config.pkl"
paths = config['paths']
model_params = config['model']['parameters']
training_params = config['training']

with open(config_path, 'rb') as f:
    config = pickle.load(f)

# ── NEW output directory to avoid overwriting ──
OUTDIR = "/rhomes/aabdel/DrosoEmbedding/results/Neuropil_Importance/v4_ols_overfit_comparison"
DATADIR = os.path.join(OUTDIR, "data")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)

# ================================================================
# STEP 1: Collect model logits on test set
# ================================================================
logger.info("=" * 60)
logger.info("STEP 1: Collecting model logits")
logger.info("=" * 60)

with open(paths["pickle_path"], 'rb') as f:
    _, _, X_test, _, _, Y_test = pickle.load(f)

X_test_original = X_test.copy() if isinstance(X_test, list) else list(X_test)

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
    raise FileNotFoundError("Trained model not found.")

classifier.eval()
all_logits, all_labels = [], []

with torch.no_grad():
    for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
        X_batch = X_batch.to(config['device'])
        output = classifier(X_batch)
        logits_batch = output[0] if isinstance(output, tuple) else output

        if batch_idx == 0:
            logger.info(f"Logits shape: {logits_batch.shape}")
            logger.info(f"Sanity: min={logits_batch.min():.3f}, max={logits_batch.max():.3f}")

        all_logits.append(logits_batch.cpu().numpy())
        all_labels.append(y_batch.numpy())

logits = np.concatenate(all_logits, axis=0)
y_true = np.concatenate(all_labels, axis=0)
N = logits.shape[0]
logger.info(f"Collected {N} samples, logits shape: {logits.shape}")

# ================================================================
# STEP 2: Build neuropil feature matrix (N x 12)
# ================================================================
logger.info("=" * 60)
logger.info("STEP 2: Building neuropil feature matrix")
logger.info("=" * 60)

neuropil_features = []

for np_idx, neuropil in enumerate(NEUROPILS):
    logger.info(f"Processing {neuropil} ({np_idx+1}/{len(NEUROPILS)})")

    config['data']['preprocessing']['neuropil'] = neuropil
    config['data']['preprocessing']['isolate_neuropil'] = True
    config['data']['preprocessing']['remove_neuropil'] = False
    setup_derived_parameters(config)
    root_np = config['paths']['allTs_path']

    X_test_np = [
        os.path.join(root_np, os.path.basename(os.path.dirname(p)), os.path.basename(p))
        for p in X_test_original
    ]

    for i in range(min(10, len(X_test_np))):
        if not os.path.exists(X_test_np[i]):
            raise FileNotFoundError(f"{neuropil}: missing {X_test_np[i]}")

    test_dataset_np = CustomDataset(
        X_test_np, Y_test, transform=True,
        seq_length=model_params['seq_len'],
        seq_steps=model_params['seq_steps'],
        allTs_path=root_np,
    )
    test_loader_np = DataLoader(
        test_dataset_np, batch_size=training_params['batch_size'],
        shuffle=False, num_workers=4, pin_memory=True,
    )

    feats = []
    y_seen = []
    for X_batch, y_batch in test_loader_np:
        if X_batch.dim() == 5:
            Xb = X_batch.mean(dim=1).squeeze(1)
        elif X_batch.dim() == 4:
            Xb = X_batch.squeeze(1)
        else:
            Xb = X_batch
        Xb = Xb.cpu().numpy().astype(np.float32)

        for i in range(Xb.shape[0]):
            nz = Xb[i][Xb[i] != 0]
            feats.append(float(nz.mean()) if nz.size > 0 else 0.0)
        y_seen.extend(y_batch.numpy().tolist())

    assert np.array_equal(np.array(y_seen), y_true), f"{neuropil}: label mismatch!"
    feats = np.array(feats, dtype=np.float32)
    neuropil_features.append(feats)
    logger.info(f"  mean={feats.mean():.6f}, std={feats.std():.6f}")

# Stack: (N, 12)
X = np.stack(neuropil_features, axis=1)
assert X.shape == (N, 12)
logger.info(f"Feature matrix: {X.shape}")

# Neuropil correlation matrix
corr_df = pd.DataFrame(X, columns=NEUROPILS).corr()
corr_df.to_csv(os.path.join(DATADIR, "neuropil_correlation.csv"))
logger.info(f"INP correlations:\n{corr_df['INP'].sort_values(ascending=False)}")

# Z-score normalization
X = X.astype(np.float64)
mu = X.mean(axis=0)
sigma = X.std(axis=0, ddof=0)
const = sigma < 1e-8
Xz = (X - mu) / (sigma + 1e-8)
Xz[:, const] = 0.0

assert np.isfinite(Xz).all(), "Non-finite values after normalization!"
assert np.max(np.abs(Xz.mean(axis=0))) < 1e-4, "Normalization failed!"
logger.info(f"Normalized. {const.sum()} constant columns.")

# ================================================================
# STEP 3: Ridge regression (neuropil activity -> logits)
# ================================================================
logger.info("=" * 60)
logger.info("STEP 3: Ridge regression")
logger.info("=" * 60)

Y = logits - logits.mean(axis=0)

alphas = np.logspace(-3, 3, 7)
ridge = RidgeCV(alphas=alphas, fit_intercept=True)
ridge.fit(Xz, Y)
logger.info(f"Selected alpha: {ridge.alpha_:.4f}")

W = ridge.coef_.T  # (12, 16)
W_signed = W.copy()
W_abs = np.abs(W)

# R2 scores
train_r2 = ridge.score(Xz, Y)
cv_scores = cross_val_score(RidgeCV(alphas=alphas, fit_intercept=True), Xz, Y, cv=5, scoring='r2')
logger.info(f"R2 (train): {train_r2:.4f}")
logger.info(f"R2 (5-fold CV): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

class_names = config['data']['classes']
Y_pred = ridge.predict(Xz)
per_class_r2 = {}
for c, cname in enumerate(class_names):
    ss_res = np.sum((Y[:, c] - Y_pred[:, c]) ** 2)
    ss_tot = np.sum((Y[:, c] - Y[:, c].mean()) ** 2)
    r2_c = 1 - ss_res / (ss_tot + 1e-8)
    per_class_r2[cname] = round(r2_c, 4)
    logger.info(f"  R2 {cname}: {r2_c:.4f}")

# ================================================================
# STEP 3b: Permutation test for R² significance
# ================================================================
logger.info("=" * 60)
logger.info("STEP 3b: Permutation test")
logger.info("=" * 60)

N_PERMUTATIONS = 1000
rng = np.random.default_rng(42)

real_r2 = train_r2

perm_r2s = []
for i in range(N_PERMUTATIONS):
    perm_idx = rng.permutation(Xz.shape[0])
    Xz_perm = Xz[perm_idx]  # shuffle X, keep Y intact

    ridge_perm = RidgeCV(alphas=alphas, fit_intercept=True)
    ridge_perm.fit(Xz_perm, Y)
    perm_r2s.append(ridge_perm.score(Xz_perm, Y))

    if (i + 1) % 100 == 0:
        logger.info(f"  Permutation {i + 1}/{N_PERMUTATIONS} (shuffle-X)")

perm_r2s = np.array(perm_r2s)
p_value = (np.sum(perm_r2s >= real_r2) + 1) / (N_PERMUTATIONS + 1)

logger.info(f"Permutation test results ({N_PERMUTATIONS} permutations):")
logger.info(f"  Real R²:       {real_r2:.4f}")
logger.info(f"  Null R² mean:  {perm_r2s.mean():.4f} +/- {perm_r2s.std():.4f}")
logger.info(f"  Null R² max:   {perm_r2s.max():.4f}")
logger.info(f"  p-value:       {p_value:.4f}")

# ================================================================
# STEP 3c: OLS and overfit comparison
# ================================================================
logger.info("=" * 60)
logger.info("STEP 3c: OLS and overfit comparison")
logger.info("=" * 60)

# --- 1. Plain OLS (no regularisation) ---
ols = LinearRegression(fit_intercept=True)
ols.fit(Xz, Y)
ols_train_r2 = ols.score(Xz, Y)
ols_cv_r2 = cross_val_score(LinearRegression(fit_intercept=True), Xz, Y, cv=5, scoring='r2').mean()
W_ols = ols.coef_.T  # (12, 16)
W_ols_abs = np.abs(W_ols)

logger.info(f"OLS  train R²: {ols_train_r2:.4f}")
logger.info(f"OLS  CV R²:    {ols_cv_r2:.4f}")

# --- 2. Overfit: polynomial features (degree=2) + OLS ---
poly = PolynomialFeatures(degree=2, include_bias=False)
Xz_poly = poly.fit_transform(Xz)   # (N, 90) — 12 linear + 78 interaction/quadratic terms
logger.info(f"Polynomial feature matrix: {Xz_poly.shape}")

ols_poly = LinearRegression(fit_intercept=True)
ols_poly.fit(Xz_poly, Y)
poly_train_r2 = ols_poly.score(Xz_poly, Y)
poly_cv_r2 = cross_val_score(LinearRegression(fit_intercept=True), Xz_poly, Y, cv=5, scoring='r2').mean()

# Extract only the 12 linear-term weights for comparability with ridge/OLS
n_linear = Xz.shape[1]  # first 12 columns of poly are the original features
W_poly_linear = ols_poly.coef_[:, :n_linear].T  # (12, 16)
W_poly_abs = np.abs(W_poly_linear)

logger.info(f"Poly train R²: {poly_train_r2:.4f}")
logger.info(f"Poly CV R²:    {poly_cv_r2:.4f}")

logger.info("\n--- R² comparison summary ---")
logger.info(f"  Ridge (regularised) : train={train_r2:.4f}  CV={cv_scores.mean():.4f}")
logger.info(f"  OLS  (no reg.)      : train={ols_train_r2:.4f}  CV={ols_cv_r2:.4f}")
logger.info(f"  Poly+OLS (overfit)  : train={poly_train_r2:.4f}  CV={poly_cv_r2:.4f}")

# --- 3. Alpha sweep: bias-variance tradeoff across regularisation strength ---
from sklearn.linear_model import Ridge as RidgePlain

alphas_sweep = np.logspace(-6, 3, 50)
sweep_train_r2s, sweep_cv_r2s = [], []

for a in alphas_sweep:
    r = RidgePlain(alpha=a, fit_intercept=True)
    r.fit(Xz, Y)
    sweep_train_r2s.append(r.score(Xz, Y))
    sweep_cv_r2s.append(
        cross_val_score(RidgePlain(alpha=a, fit_intercept=True),
                        Xz, Y, cv=5, scoring='r2').mean()
    )

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.semilogx(alphas_sweep, sweep_train_r2s, label='Train R²', color='steelblue', linewidth=2)
ax.semilogx(alphas_sweep, sweep_cv_r2s,   label='CV R²',    color='coral',     linewidth=2)
ax.axvline(ridge.alpha_, color='black', linestyle='--', linewidth=1.2,
           label=f'RidgeCV selected α = {ridge.alpha_:.4f}')

# Mark OLS (α=0) as a star on the left y-axis
ax.plot(alphas_sweep[0], ols_train_r2, marker='*', color='steelblue', markersize=12, zorder=5)
ax.plot(alphas_sweep[0], ols_cv_r2,   marker='*', color='coral',     markersize=12, zorder=5)
ax.annotate(f'α = 0\n(OLS / LinearRegression)\ntrain={ols_train_r2:.4f}, CV={ols_cv_r2:.4f}',
            xy=(alphas_sweep[0], ols_cv_r2),
            xytext=(alphas_sweep[3], ols_cv_r2 - 0.006),
            fontsize=FONT_SIZES['annotation'], color='#555555',
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1))

ax.set_xlabel('α (regularisation strength)', fontsize=FONT_SIZES['label'])
ax.set_ylabel('R²', fontsize=FONT_SIZES['label'])
ax.set_title('Bias-variance tradeoff: Ridge regularisation sweep', fontsize=FONT_SIZES['subplot_title'])
ax.legend(fontsize=FONT_SIZES['legend'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "alpha_sweep.png"), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(OUTDIR, "alpha_sweep.pdf"), format='pdf', bbox_inches='tight')
plt.close()
logger.info("Saved alpha_sweep.png/pdf")

# --- 3. Contrast plots: compare the three methods side by side ---
contrast_keys = {
    'Modality (Odor - Taste)':        'Odor_minus_Taste',
    'Valence (App. - Avers.)':        'Appetitive_minus_Aversive',
    'State (Starved - Fed)':          'Starved_minus_Fed',
}

groups = {
    "Odor":       [0, 1, 2, 3],
    "Taste":      [4, 5, 6, 7],
    "Multi":      [8, 9, 10, 11, 12, 13, 14, 15],
    "Appetitive": [0, 2, 4, 6, 8, 12],
    "Aversive":   [1, 3, 5, 7, 9, 13],
    "Conflicting":[10, 11, 14, 15],
    "Starved":    [0, 1, 4, 5, 8, 9, 10, 11],
    "Fed":        [2, 3, 6, 7, 12, 13, 14, 15],
}
contrasts = {
    "Odor_minus_Taste":          ("Odor", "Taste"),
    "Appetitive_minus_Aversive": ("Appetitive", "Aversive"),
    "Starved_minus_Fed":         ("Starved", "Fed"),
}

def compute_contrast_abs(W_abs_mat, groups, contrasts):
    ga = {name: W_abs_mat[:, idx].mean(axis=1) for name, idx in groups.items()}
    return {cname: ga[g1] - ga[g2] for cname, (g1, g2) in contrasts.items()}

contrast_ridge = compute_contrast_abs(W_abs,     groups, contrasts)
contrast_ols   = compute_contrast_abs(W_ols_abs,  groups, contrasts)
contrast_poly  = compute_contrast_abs(W_poly_abs, groups, contrasts)

fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharey='row')
method_labels = [
    f'Ridge (train R²={train_r2:.3f}, CV={cv_scores.mean():.3f})',
    f'OLS   (train R²={ols_train_r2:.3f}, CV={ols_cv_r2:.3f})',
    f'Poly+OLS (train R²={poly_train_r2:.3f}, CV={poly_cv_r2:.3f})',
]
contrast_data = [contrast_ridge, contrast_ols, contrast_poly]

# Colors by factor (row), consistent with fig4_contrasts
factor_colors = {
    'Odor_minus_Taste':          ('#D35F2A', '#3382BE'),   # orange / blue
    'Appetitive_minus_Aversive': ('#2CA02C', '#D62728'),   # green  / red
    'Starved_minus_Fed':         ('#808080', '#404040'),   # light grey / dark grey
}
contrast_ckeys = list(contrast_keys.values())

for col, (method_label, cdata) in enumerate(zip(method_labels, contrast_data)):
    for row, (title, ckey) in enumerate(contrast_keys.items()):
        ax = axes[row, col]
        vals = cdata[ckey]
        cp, cn = factor_colors[ckey]
        bar_colors = [cp if v >= 0 else cn for v in vals]
        ax.bar(range(len(NEUROPILS)), vals, color=bar_colors)
        ax.set_xticks(range(len(NEUROPILS)))
        ax.set_xticklabels(NEUROPILS, rotation=45, ha='right', fontsize=FONT_SIZES['tick'])
        ax.axhline(0, color='black', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if col == 0:
            ax.set_ylabel(f'Δ |W|\n{title}', fontsize=FONT_SIZES['label'])
        if row == 0:
            ax.set_title(method_label, fontsize=FONT_SIZES['subplot_title'], fontweight='bold')

plt.suptitle('Neuropil contrast comparison: Ridge vs OLS vs Polynomial overfit', fontsize=FONT_SIZES['title'], y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "comparison_ridge_ols_poly.png"), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(OUTDIR, "comparison_ridge_ols_poly.pdf"), format='pdf', bbox_inches='tight')
plt.close()
logger.info("Saved comparison_ridge_ols_poly.png/pdf")

# Save comparison summary
comparison_summary = {
    "ridge": {"train_r2": round(train_r2, 4), "cv_r2": round(cv_scores.mean(), 4)},
    "ols":   {"train_r2": round(ols_train_r2, 4), "cv_r2": round(ols_cv_r2, 4)},
    "poly_ols": {"train_r2": round(poly_train_r2, 4), "cv_r2": round(poly_cv_r2, 4),
                 "n_features": int(Xz_poly.shape[1])},
}
with open(os.path.join(DATADIR, "comparison_summary.json"), 'w') as f:
    json.dump(comparison_summary, f, indent=2)
logger.info("Saved comparison_summary.json")

# ================================================================
# Save data
# ================================================================

np.save(os.path.join(DATADIR, "W_abs.npy"), W_abs)
np.save(os.path.join(DATADIR, "W_signed.npy"), W_signed)

df_abs = pd.DataFrame(W_abs.T, index=class_names, columns=NEUROPILS)
df_abs.to_csv(os.path.join(DATADIR, "W_abs.csv"))

with open(os.path.join(DATADIR, "neuropil_names.json"), 'w') as f:
    json.dump(NEUROPILS, f)

ridge_summary = {
    "alpha": float(ridge.alpha_),
    "r2_train": round(train_r2, 4),
    "r2_cv_mean": round(cv_scores.mean(), 4),
    "r2_cv_std": round(cv_scores.std(), 4),
    "r2_cv_folds": [round(s, 4) for s in cv_scores],
    "r2_per_class": per_class_r2,
    "permutation_p_value": round(p_value, 4),
    "permutation_null_r2_mean": round(perm_r2s.mean(), 4),
    "permutation_null_r2_std": round(perm_r2s.std(), 4),
    "permutation_null_r2_max": round(perm_r2s.max(), 4),
    "permutation_n": N_PERMUTATIONS,
}
with open(os.path.join(DATADIR, "ridge_summary.json"), 'w') as f:
    json.dump(ridge_summary, f, indent=2)

logger.info("Saved weights and summary to data/")

logger.info("\nTop 3 neuropils per class (|W|):")
for c, cname in enumerate(class_names):
    top3 = np.argsort(W_abs[:, c])[-3:][::-1]
    logger.info(f"  {cname}: {[(NEUROPILS[i], f'{W_abs[i,c]:.3f}') for i in top3]}")

# ================================================================
# STEP 4: Group analysis and contrasts
# ================================================================
logger.info("=" * 60)
logger.info("STEP 4: Group analysis and contrasts")
logger.info("=" * 60)

groups = {
    "Odor":       [0, 1, 2, 3],
    "Taste":      [4, 5, 6, 7],
    "Multi":      [8, 9, 10, 11, 12, 13, 14, 15],
    "Appetitive": [0, 2, 4, 6, 8, 12],
    "Aversive":   [1, 3, 5, 7, 9, 13],
    "Conflicting":[10, 11, 14, 15],
    "Starved":    [0, 1, 4, 5, 8, 9, 10, 11],
    "Fed":        [2, 3, 6, 7, 12, 13, 14, 15],
}

group_abs = {name: W_abs[:, idx].mean(axis=1) for name, idx in groups.items()}

df_group_abs = pd.DataFrame(group_abs, index=NEUROPILS).T
df_group_abs.to_csv(os.path.join(DATADIR, "group_profiles_abs.csv"))

contrasts = {
    "Odor_minus_Taste":          ("Odor", "Taste"),
    "Appetitive_minus_Aversive": ("Appetitive", "Aversive"),
    "Starved_minus_Fed":         ("Starved", "Fed"),
}

contrast_abs = {}
for cname, (g1, g2) in contrasts.items():
    contrast_abs[cname] = group_abs[g1] - group_abs[g2]

df_contrast = pd.DataFrame(contrast_abs, index=NEUROPILS).T
df_contrast.to_csv(os.path.join(DATADIR, "group_contrasts_abs.csv"))

logger.info("\nGroup top 3 (|W|):")
for gname, profile in group_abs.items():
    top3 = np.argsort(profile)[-3:][::-1]
    logger.info(f"  {gname}: {[(NEUROPILS[i], f'{profile[i]:.3f}') for i in top3]}")

logger.info("\nContrast top 3 pos/neg:")
for cname, delta in contrast_abs.items():
    top_pos = np.argsort(delta)[-3:][::-1]
    top_neg = np.argsort(delta)[:3]
    logger.info(f"  {cname}:")
    logger.info(f"    + {[(NEUROPILS[i], f'{delta[i]:.3f}') for i in top_pos]}")
    logger.info(f"    - {[(NEUROPILS[i], f'{delta[i]:.3f}') for i in top_neg]}")

# ================================================================
# STEP 5: Figures
# ================================================================
logger.info("=" * 60)
logger.info("STEP 5: Generating figures")
logger.info("=" * 60)

# 1. Heatmap: all classes x neuropils
fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(df_abs.values, aspect='auto', cmap='viridis')
ax.set_xticks(range(len(NEUROPILS)))
ax.set_xticklabels(NEUROPILS, rotation=45, ha='right')
ax.set_yticks(range(len(class_names)))
ax.set_yticklabels(class_names)
ax.set_xlabel("Neuropil"); ax.set_ylabel("Class")
ax.set_title("Ridge Weights (|W|): Neuropil -> Class Logits")
plt.colorbar(im, ax=ax)
for i in range(len(class_names)):
    for j in range(len(NEUROPILS)):
        ax.text(j, i, f"{df_abs.values[i,j]:.2f}", ha='center', va='center', fontsize=FONT_SIZES['heatmap_cell'], color='white')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "heatmap_classes_abs.png"), dpi=150)
plt.close()
logger.info("Saved heatmap_classes_abs.png")

# 2. Heatmap: groups x neuropils
fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(df_group_abs.values, aspect='auto', cmap='viridis')
ax.set_xticks(range(len(NEUROPILS)))
ax.set_xticklabels(NEUROPILS, rotation=45, ha='right')
ax.set_yticks(range(len(df_group_abs)))
ax.set_yticklabels(df_group_abs.index)
ax.set_xlabel("Neuropil"); ax.set_ylabel("Group")
ax.set_title("Group Profiles (|W|)")
plt.colorbar(im, ax=ax)
for i in range(df_group_abs.shape[0]):
    for j in range(df_group_abs.shape[1]):
        ax.text(j, i, f"{df_group_abs.values[i,j]:.2f}", ha='center', va='center', fontsize=FONT_SIZES['heatmap_cell'], color='white')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "heatmap_groups_abs.png"), dpi=150)
plt.close()
logger.info("Saved heatmap_groups_abs.png")

# 3. Contrast barplots (3-panel, main figure)
fig4_spec = {
    'Modality': {
        'key': 'Odor_minus_Taste',
        'pos_label': 'Odor', 'neg_label': 'Taste',
        'pos_color': '#D35F2A', 'neg_color': '#3382BE',
    },
    'Valence': {
        'key': 'Appetitive_minus_Aversive',
        'pos_label': 'Appetitive', 'neg_label': 'Aversive',
        'pos_color': '#2CA02C', 'neg_color': '#D62728',
    },
    'State': {
        'key': 'Starved_minus_Fed',
        'pos_label': 'Starved', 'neg_label': 'Fed',
        'pos_color': '#808080', 'neg_color': '#404040',
    },
}

all_vals = np.concatenate([contrast_abs[info['key']] for info in fig4_spec.values()])
ylim_max = np.abs(all_vals).max() * 1.15

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, (title, info) in zip(axes, fig4_spec.items()):
    vals = contrast_abs[info['key']]
    colors = [info['pos_color'] if v >= 0 else info['neg_color'] for v in vals]
    ax.bar(range(len(NEUROPILS)), vals, color=colors)
    ax.set_xticks(range(len(NEUROPILS)))
    ax.set_xticklabels(NEUROPILS, rotation=45, ha='right', fontsize=FONT_SIZES['tick'])
    ax.set_title(title, fontsize=FONT_SIZES['subplot_title'], fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylim(-ylim_max, ylim_max)
    if ax == axes[0]:
        ax.set_ylabel('Δ |W|', fontsize=FONT_SIZES['label'])
    ax.legend(handles=[
        Patch(facecolor=info['pos_color'], label=info['pos_label']),
        Patch(facecolor=info['neg_color'], label=info['neg_label']),
    ], loc='upper right', fontsize=FONT_SIZES['legend'], framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig4_contrasts_3panel.png"), dpi=300)
plt.close()
logger.info("Saved fig4_contrasts_3panel.png")

# 4. Summary barplots per factor
summary_specs = {
    "Modality": {"Odor": [0,1,2,3], "Taste": [4,5,6,7], "Multi": [8,9,10,11,12,13,14,15]},
    "State":    {"Starved": [0,1,4,5,8,9,10,11], "Fed": [2,3,6,7,12,13,14,15]},
    "Valence":  {"Appetitive": [0,2,4,6,8,12], "Aversive": [1,3,5,7,9,13], "Mixed": [10,11,14,15]},
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
x = np.arange(len(NEUROPILS))
for ax, (title, group_dict) in zip(axes, summary_specs.items()):
    profiles = np.stack([W_abs[:, idx].mean(axis=1) for idx in group_dict.values()])
    agg = profiles.mean(axis=0)
    sort_idx = np.argsort(agg)[::-1]
    ax.bar(x, agg[sort_idx], color='steelblue')
    ax.set_title(title, fontsize=FONT_SIZES['subplot_title'], fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([NEUROPILS[i] for i in sort_idx], rotation=45, ha='right')
    if ax == axes[0]:
        ax.set_ylabel("Mean |W|", fontsize=FONT_SIZES['label'])
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "summary_barplots_factors.png"), dpi=150)
plt.close()
logger.info("Saved summary_barplots_factors.png")

# 5. Overall sorted barplot
overall = W_abs.mean(axis=1)
sort_idx = np.argsort(overall)[::-1]
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(NEUROPILS)), overall[sort_idx], color='steelblue')
ax.set_xticks(range(len(NEUROPILS)))
ax.set_xticklabels([NEUROPILS[i] for i in sort_idx], rotation=45, ha='right')
ax.set_xlabel("Neuropil"); ax.set_ylabel("Mean |W|")
ax.set_title("Overall Neuropil Importance (sorted)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "summary_barplot_overall.png"), dpi=150)
plt.close()
logger.info("Saved summary_barplot_overall.png")

# 6. Neuropil correlation matrix
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_df.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(NEUROPILS)))
ax.set_xticklabels(NEUROPILS, rotation=45, ha='right')
ax.set_yticks(range(len(NEUROPILS)))
ax.set_yticklabels(NEUROPILS)
ax.set_title("Inter-neuropil correlation (mean activity)")
plt.colorbar(im, ax=ax, shrink=0.8)
for i in range(len(NEUROPILS)):
    for j in range(len(NEUROPILS)):
        ax.text(j, i, f"{corr_df.values[i,j]:.2f}", ha='center', va='center', fontsize=FONT_SIZES['heatmap_cell'])
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "correlation_matrix.png"), dpi=150)
plt.close()
logger.info("Saved correlation_matrix.png")

# 7. Permutation test histogram
ig, ax = plt.subplots(figsize=(7, 4.5))
 
# Main histogram: full range including real R²
bins_main = np.linspace(0, max(real_r2 * 1.15, perm_r2s.max() * 1.15), 60)
ax.hist(perm_r2s, bins=bins_main, color='#7f8c8d', alpha=0.8, edgecolor='white', linewidth=0.5,
        label=f'Null distribution (n={N_PERMUTATIONS})')
ax.axvline(real_r2, color='#c0392b', linewidth=2, linestyle='--',
           label=f'Observed $R^2$ = {real_r2:.3f}')
 
ax.set_xlabel('$R^2$', fontsize=FONT_SIZES['label'])
ax.set_ylabel('Count', fontsize=FONT_SIZES['label'])
ax.set_title(f'Permutation test ($p$ < {1/N_PERMUTATIONS:.3f})', fontsize=FONT_SIZES['subplot_title'])
ax.legend(fontsize=FONT_SIZES['legend'], loc='upper center')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
 
# Inset: zoomed view of the null distribution
ax_inset = inset_axes(ax, width="45%", height="55%", loc='upper right',
                      bbox_to_anchor=(0.0, -0.02, 1, 1), bbox_transform=ax.transAxes)
 
bins_inset = np.linspace(perm_r2s.min() - 0.0001, perm_r2s.max() * 1.3, 40)
ax_inset.hist(perm_r2s, bins=bins_inset, color='#7f8c8d', alpha=0.8, edgecolor='white', linewidth=0.5)
ax_inset.set_xlabel('$R^2$', fontsize=FONT_SIZES['label'])
ax_inset.set_ylabel('Count', fontsize=FONT_SIZES['label'])
ax_inset.set_title('Null distribution (zoom)', fontsize=FONT_SIZES['subplot_title'])
ax_inset.tick_params(labelsize=FONT_SIZES['tick'])
ax_inset.spines['top'].set_visible(False)
ax_inset.spines['right'].set_visible(False)
 
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "permutation_test_histogram.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTDIR, "permutation_test_histogram.pdf"), format='pdf', bbox_inches='tight')
plt.close()
logger.info("Saved permutation_test_histogram.png/pdf")
 

logger.info("=" * 60)
logger.info("DONE.")
logger.info(f"  Figures: {OUTDIR}/")
logger.info(f"  Data:    {DATADIR}/")
logger.info("=" * 60)