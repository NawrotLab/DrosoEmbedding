import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
import torch
import pickle

from src.models.cnn_transformer import CNN_Transformer
from src.utils.config_loader import load_config
from src.data.dataset import CustomDataset
from torch.utils.data import DataLoader
from src.utils.helpers import get_predictions
from sklearn.metrics import confusion_matrix

# — setup paths —
config            = load_config()
device            = config['device']
base_results_path = '/rhomes/aabdel/DrosoEmbedding/results'
viz_path          = os.path.join(base_results_path, 'CombiPlots')
os.makedirs(viz_path, exist_ok=True)

pickle_path = '/rhomes/aabdel/DrosoEmbedding/pickles/TrainValTest_LocalScratch_Paths-Labels/WithinAnimal'
allTs_path  = "/localscratch/aabdel/imgs4DL/meanZ_allTs"

# File lists
model_paths = [
    os.path.join(base_results_path,
        'MetabolicState_2_goBinary_1kEps_newLR_2/models',
        'CNN_Transformer_nr_channels=1_embed_dim=16_num_heads=2_'
        'num_layers=1_dropout=0.3_seq_len=5_seq_steps=10_nr_classes=2.pth'),
    os.path.join(base_results_path,
        'State_Modality_6_go6_2/models',
        'CNN_Transformer_nr_channels=1_embed_dim=16_num_heads=2_'
        'num_layers=1_dropout=0.3_seq_len=5_seq_steps=10_nr_classes=6.pth'),
    os.path.join(base_results_path,
        'State_Modality_Valence_16_go16_3kEps_2/models',
        'CNN_Transformer_nr_channels=1_embed_dim=16_num_heads=2_'
        'num_layers=1_dropout=0.3_seq_len=5_seq_steps=10_nr_classes=16.pth')
]

pickles = [
    os.path.join(pickle_path, 'meanZ_logTs_S_F.pickle'),
    os.path.join(pickle_path, 'meanZ_logTs_SO_FO_ST_FT_SM_FM.pickle'),
    os.path.join(pickle_path, 'meanZ_logTs_SOP_SON_FOP_FON_STP_STN_FTP_FTN_SMMP_SMMN_SMCP_SMCN_FMMP_FMMN_FMCP_FMCN.pickle')
]

row_titles = ['2‑Class', '6‑Class', '16‑Class']
# Define dummy class names; replace with your real labels
classes_list = [
    [f"Class {j}" for j in range(2)],
    [f"Class {j}" for j in range(6)],
    [f"Class {j}" for j in range(16)],
]

# Create grid for confusion matrices
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(25, 15))
fig.suptitle("Confusion Matrices", fontsize=20)

for i, (pth, pkl, title) in enumerate(zip(model_paths, pickles, row_titles)):
    # 1) Load model
    if not os.path.exists(pth):
        raise FileNotFoundError(f"Checkpoint not found: {pth}")
    ckpt = torch.load(pth, map_location=device)
    params = ckpt.get('params', {})
    model  = CNN_Transformer(**params).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    # 2) Load data
    with open(pkl, 'rb') as f:
        _, _, X_test, _, _, Y_test = pickle.load(f)
    ds = CustomDataset(X_test, Y_test, transform=True,
                       seq_length=5, seq_steps=10, allTs_path=allTs_path)
    loader = DataLoader(ds, batch_size=256, shuffle=False,
                        num_workers=4, pin_memory=True)

    # 3) Predictions
    _, y_true = get_predictions(model, loader, device)
    y_pred, _ = get_predictions(model, loader, device)

    # 4) Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # 5) Inline plotting into col 0
    ax = axs[i, 3]
    # Compute percentages
    with np.errstate(all='ignore'):
        cm_pct = (cm / cm.sum(axis=1, keepdims=True)) * 100
        cm_pct = np.nan_to_num(cm_pct)
    # Annotate labels
    annot = np.array([
        [f"{pct:.1f}%\n({cnt})" for pct, cnt in zip(row_pct, row)]
        for row_pct, row in zip(cm_pct, cm)
    ])
    # Plot heatmap
    sns.heatmap(
        cm_pct,
        # annot=annot,
        fmt="",
        cmap='Blues',
        xticklabels=classes_list[i],
        yticklabels=classes_list[i],
        cbar=False,
        vmin=0, vmax=100,
        annot_kws={"size": 15 if len(classes_list[i])>=4 else 30},
        ax=ax
    )
    ax.set_title(f"{title} (Acc: {cm.trace()/cm.sum()*100:.1f}% )")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=90)

    # Blank out other cols
    for j in range(1,4):
        axs[i, j].axis('off')

# Save combined figure
out_file = os.path.join(viz_path, 'all_confusion_matrices.png')
plt.tight_layout(rect=[0,0.03,1,0.95])
fig.savefig(out_file, dpi=150)
plt.close(fig)
print(f"Saved combined image as {out_file}")
