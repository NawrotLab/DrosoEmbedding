import os, re, pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines


# --- CONFIG ---
RESULTS_DIR = Path("results/_CNN-TRF_new").expanduser().resolve()
print(f"[info] RESULTS_DIR = {RESULTS_DIR}")

prefixes = ["C2_E16_1K", "C6_E16_1K", "C16_E16_1K"]


# --- HELPERS ---
def moving_average(x, window_size=3):
    return np.convolve(x, np.ones(window_size)/window_size, mode="valid")

_nat = lambda p: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]

def load_runs(prefix: str):
    files = sorted(RESULTS_DIR.glob(f"{prefix}_*.pkl"), key=_nat)
    print(f"[load] {prefix}: {len(files)} files -> {[f.name for f in files]}")
    runs = []
    for f in files:
        with open(f, "rb") as fh:
            runs.append(pickle.load(fh))
    print(runs)
    return runs
def plot_runs(ax, runs, title, cmap_name="tab10"):
    ax2 = ax.twinx()
    epochs_range = lambda run: range(1, len(run["train_loss"]) + 1)

    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i % cmap.N) for i in range(len(runs))]

    handles, labels = [], []
    for k, run in enumerate(runs):
        ep = epochs_range(run)

        # --- grey training curves (only once in legend) ---
        if k == 0:
            h1, = ax.plot(ep, run["train_loss"], "-", color="grey", linewidth=1, alpha=0.5, label="Train Loss")
            h2, = ax2.plot(ep, run["train_acc"], "--", color="grey", linewidth=1, alpha=0.5, label="Train Acc")
            handles.extend([h1, h2]); labels.extend([h1.get_label(), h2.get_label()])
        else:
            ax.plot(ep, run["train_loss"], "-", color="grey", linewidth=1, alpha=0.5)
            ax2.plot(ep, run["train_acc"], "--", color="grey", linewidth=1, alpha=0.5)

        # --- validation curves ---
        sm = moving_average(run["val_loss"])
        ep_sm = range(1, len(sm) + 1)
        h3, = ax.plot(ep_sm, sm, "-", color=colors[k], linewidth=1.2, alpha=0.8, label=f"Run {k+1} Val Loss")
        h4, = ax2.plot(ep, run["val_acc"], "--", color=colors[k], linewidth=1.2, alpha=0.8, label=f"Run {k+1} Val Acc")
        handles.extend([h3, h4]); labels.extend([h3.get_label(), h4.get_label()])
    
    ax.grid(False)         
    ax2.grid(True, which="both", axis="both", linestyle="--", alpha=0.6)
    ax.patch.set_visible(False)      # or: ax.set_facecolor('none')
    ax2.set_axisbelow(True)          # grid below lines on ax2
    ax.set_zorder(2); ax2.set_zorder(1)
    ax2.set_axisbelow(True)
    ax.set_xlim(1, 1000)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross Entropy Loss")
    ax2.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=11)
    return handles, labels


# --- LOAD + PLOT (3 ROWS) ---
runs_list = [load_runs(p) for p in prefixes]
# sanity: stop early if any prefix has no files
for p, r in zip(prefixes, runs_list):
    if not r:
        raise FileNotFoundError(f"No runs found for prefix '{p}'. Check RESULTS_DIR/prefix.")

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

for ax, prefix, runs in zip(axes, prefixes, runs_list):
    handles, labels = plot_runs(ax, runs, title=prefix)
    if prefix == "C2_E16_1K":
        ax.set_ylim(0.1, 1)
    elif prefix == "C6_E16_1K":
        ax.set_ylim(0.4, 1.8)
    elif prefix == "C16_E16_1K":
        ax.set_ylim(0.5, 3)


fig.legend(handles, labels, loc="upper right",  bbox_to_anchor=(1.03, 0.9), frameon=False, fontsize=12)
plt.tight_layout(rect=[0,0,0.85,1])
fig.savefig("results/CombiPlots/newCNNTRF_loss_accuracies.png", dpi=300, bbox_inches="tight")
print("[done] saved.")