import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines


# --- CONFIG ---
RESULTS_DIR = Path("results/Adam_Smoothing_Runs/_ResultsSummary")   # <- change me
save_path   = RESULTS_DIR / "grid_clean.png"

rows = ["C2", "C6", "C16"]
cols = [
    ("AdamW",          "AdamW"),
    ("AdamW_Decay001", "AdamW_Decay001"),
    ("LabelSmoothing", "LabelSmoothing"),
    ("AdamW001_SmoothLabel", "AdamW001_SmoothLabel")

]

# --- HELPERS ---

def moving_average(x, w=5):
    """Simple moving average with window size w"""
    smoothed = np.convolve(x, np.ones(w)/w, mode='valid')
    pad_left = np.full((w//2,), smoothed[0])     # repeat first value
    pad_right = np.full((len(x) - len(smoothed) - len(pad_left),), smoothed[-1])
    return np.concatenate((pad_left, smoothed, pad_right))

def load_runs(prefix: str):
    runs = []
    for i in (1, 2):
        f = RESULTS_DIR / f"{prefix}_{i}.pkl"
        if f.exists():
            with open(f, "rb") as fh:
                runs.append(pickle.load(fh))
    return runs
def plot_condition(ax, pre_runs, other_runs, title):
    ax2 = ax.twinx()
    epochs_range = lambda run: range(1, len(run["train_loss"]) + 1)

    handles, labels = [], []

    # Grey (train) comes first
    for runs in (pre_runs, other_runs):
        for run in runs:
            ep = epochs_range(run)
            h1, = ax.plot(ep, run["train_loss"], "-",  color="lightgrey", linewidth=1, alpha=0.8,
                          label="Train Loss")
            h2, = ax2.plot(ep, run["train_acc"],  "--", color="lightgrey", linewidth=1, alpha=0.8,
                           label="Train Acc")
            handles.extend([h1, h2]); labels.extend([h1.get_label(), h2.get_label()])

    # PreChange: dark/light blue
    pre_colors = ["navy", "royalblue"]
    for k, run in enumerate(pre_runs):
        ep = epochs_range(run)
        smoothed = moving_average(run["val_loss"])
        h1, = ax.plot(ep, smoothed, "-",  color=pre_colors[k % 2], linewidth=2,
                      label=f"PreChange r{k+1} Loss")
        h2, = ax2.plot(ep, run["val_acc"], "--", color=pre_colors[k % 2], linewidth=2,
                       label=f"PreChange r{k+1} Acc")
        handles.extend([h1, h2]); labels.extend([h1.get_label(), h2.get_label()])

    # Variant: dark/light red
    other_colors = ["darkred", "indianred"]
    for k, run in enumerate(other_runs):
        ep = epochs_range(run)
        smoothed = moving_average(run["val_loss"])
        h1, = ax.plot(ep, smoothed, "-",  color=other_colors[k % 2], linewidth=2,
                      label=f"Variant r{k+1} Loss")
        h2, = ax2.plot(ep, run["val_acc"], "--", color=other_colors[k % 2], linewidth=2,
                       label=f"Variant r{k+1} Acc")
        handles.extend([h1, h2]); labels.extend([h1.get_label(), h2.get_label()])

    ax.set_xlabel("Epoch")
    ax.set_xlim(1, 200)
    # ax.set_ylim(0, 1.6)
    ax.set_ylabel("Cross Entropy Loss")
    ax2.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=11)

    return handles, labels

# --- FIGURE ---
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 12), constrained_layout=True)

legend_handles, legend_labels = [], []

for i, cls in enumerate(rows):
    for j, (col_label, other_tag) in enumerate(cols):
        pre_runs   = load_runs(f"{cls}_PreChange")
        other_runs = load_runs(f"{cls}_{other_tag}")
        title = f"{cls}: PreChange vs {col_label}"
        ax = axes[i, j]
        if pre_runs or other_runs:
            h, l = plot_condition(ax, pre_runs, other_runs, title)
            if not legend_handles:  # only take once (from first subplot)
                legend_handles, legend_labels = h, l
        else:
            ax.set_title(title + " (no files found)")
        
        if cls == "C2":
            axes[i,j].set_ylim(0, 1.6)
        elif cls == "C6":
            axes[i,j].set_ylim(0, 1.75)
        elif cls == "C16":
            axes[i,j].set_ylim(0, 3.5)
    print(f"Done with {cls}")

# Column headers
axes[0,0].set_title("PreChange vs AdamW", fontsize=12)
axes[0,1].set_title("PreChange vs AdamW_Decay001", fontsize=12)
axes[0,2].set_title("PreChange vs LabelSmoothing", fontsize=12)
axes[0,3].set_title("PreChange vs AdamW001_LabelSmoothing", fontsize=12)

# Row labels
for i, cls in enumerate(rows):
    axes[i,0].text(-0.25, 0.5, cls, transform=axes[i,0].transAxes,
                   rotation=90, va="center", ha="center", fontsize=12)

# Single legend for whole fig
fig.legend(
    legend_handles,
    legend_labels,
    fontsize=10,
    loc="lower center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, -0.08)   # push slightly above the axes area
)

plt.subplots_adjust(bottom=0.2)  # add extra space at bottom so legend fits
plt.suptitle("Training/Validation Loss & Accuracy (2 runs each)", y=1.02, fontsize=16)

plt.savefig("results/CombiPlots/Combi_loss_accuracies_4.png", dpi=400, bbox_inches="tight")
print("Saved..")

# --- SECOND FIGURE: C2 COMPARISON ---
c2_variants = {
    "C2_PreChange": "navy",
    "C2_LabelSmoothing": "darkred",
    "C2_AdamW001_SmoothLabel": "darkgreen",
    "C2_AdamW001_SmoothLabel1": "purple",
    "C2_AdamW001_SmoothLabel05_lr001": "orange",
}

fig2, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), constrained_layout=True)
ax_loss, ax_acc = axes


for prefix, color in c2_variants.items():
    runs = load_runs(prefix)
    for run in runs:
        epochs = range(1, len(run["val_loss"]) + 1)
        smoothed_loss = moving_average(run["val_loss"],)
        smoothed_acc = moving_average(run["val_acc"], 20)
        ax_loss.plot(epochs, smoothed_loss, "-", color=color, alpha=0.8, linewidth=2)
        ax_acc.plot(epochs, smoothed_acc, "-", color=color, alpha=0.8, linewidth=2)

# Labels & layout
ax_loss.set_xlabel("  ")
ax_loss.set_ylabel("Cross Entropy Loss")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_ylim(85, 94)
ax_loss.set_xlim(1, 200)
ax_acc.set_xlim(1, 200)
ax_loss.set_ylim(0.2, 1)   # match your C2 scale
ax_loss.set_title("C2: Loss Comparison", fontsize=14)
ax_acc.set_title("C2: Accuracy Comparison", fontsize=14)

# Legend: only one entry per variant
handles = []
labels = []
for prefix, color in c2_variants.items():
    handles.append(ax.plot([], [], color=color, lw=2)[0])
    labels.append(prefix.replace("C2_", ""))  # shorter name in legend

fig2.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
plt.subplots_adjust(bottom=0.18)

plt.savefig("results/CombiPlots/C2_all_variants_2.png", dpi=400, bbox_inches="tight")
print("Saved C2 comparison figure..")

# --- THIRD FIGURE: C6 COMPARISON ---
# --- THIRD FIGURE: C2/C6/C16 variant vs dropout (loss+acc per row) ---

# Variants (same colors across tasks)
c2_variants = {
    "C2_AdamW001_SmoothLabel05_lr001": "navy",
    "C2_AdamW0001_SmoothLabel_lr001_do1": "orange",
}
c6_variants = {
    "C6_AdamW001_SmoothLabel_lr001_do3": "navy",
    "C6_AdamW0001_SmoothLabel_lr001_do1": "orange",
}
c16_variants = {
    "C16_AdamW001_SmoothLabel_lr001_do3": "navy",
    "C16_AdamW0001_SmoothLabel_lr001_do1": "orange",
}


def plot_task_row(ax, variants: dict, title: str,
                  xlim=(1, 200), loss_ylim=None, acc_ylim=None,
                  acc_window=20, plot_Train = True):
    """Plot one row: train vs val for loss (left) and accuracy (right)."""
    ax_r = ax.twinx()
    for prefix, color in variants.items():
        runs = load_runs(prefix)
        for run in runs:
            epochs = range(1, len(run["train_loss"]) + 1)

            # Val loss (light color, solid)
            ax.plot(epochs, moving_average(run["val_loss"]), "-", color=color, alpha=1, linewidth=2, label=f"{prefix} val loss")

            # Val acc (light color, dashed)
            ax_r.plot(epochs, moving_average(run["val_acc"], acc_window), "--", color=color, alpha=1, linewidth=2, label=f"{prefix} val acc")

            if plot_Train:
                # Train loss (dark color, solid)
                ax.plot(epochs,
                        moving_average(run["train_loss"]),
                        "-", color=color, alpha=0.5, linewidth=2,
                        label=f"{prefix} train loss")

                # Train acc (dark color, dashed)
                ax_r.plot(epochs,
                        moving_average(run["train_acc"], acc_window),
                        "--", color=color, alpha=0.5, linewidth=2,
                        label=f"{prefix} train acc")

    ax.set_xlim(*xlim)
    if loss_ylim: ax.set_ylim(*loss_ylim)
    if acc_ylim:  ax_r.set_ylim(*acc_ylim)

    ax.set_ylabel("Cross Entropy Loss")
    ax_r.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.25)


# Build figure (3 rows, 1 col)
fig3, (ax_c2, ax_c6, ax_c16) = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), constrained_layout=True)

# Per-row plotting (adjust y-lims to your scales; comment/remove if you prefer autoscale)
plot_task_row(ax_c2, c2_variants, "C2",
              xlim=(1, 200),
              loss_ylim=(0.1, 1),   # your earlier note for C2-ish scale
              acc_ylim=(70, 102))          # set e.g. (0.85, 0.94) if your acc is 0–1; or (85,94) if in %
plot_task_row(ax_c6, c6_variants, "C6",
              xlim=(1, 200),
              loss_ylim=(0.3, 1.6),
              acc_ylim=(70, 102))
plot_task_row(ax_c16, c16_variants, "C16",
              xlim=(1, 200),
              loss_ylim=(0.4, 3.0),
              acc_ylim=(20, 102))

# One shared legend (use C2 keys for short labels)
# handles = [plt.Line2D([], [], color=col, lw=2) for col in c2_variants.values()]
# labels  = [k.replace("C2_", "") for k in c2_variants.keys()]
# fig3.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))

# Create custom legend entries
handles = []

# For each variant color
for prefix, color in c2_variants.items():
    # Train loss (dark solid)
    handles.append(mlines.Line2D([], [], color=color, linestyle='-', lw=2, alpha=0.9,
                                 label=f"{prefix.replace('C2_', '')} train loss"))
    # Val loss (light solid)
    handles.append(mlines.Line2D([], [], color=color, linestyle='-', lw=2, alpha=0.5,
                                 label=f"{prefix.replace('C2_', '')} val loss"))
    # Train acc (dark dashed)
    handles.append(mlines.Line2D([], [], color=color, linestyle='--', lw=2, alpha=0.9,
                                 label=f"{prefix.replace('C2_', '')} train acc"))
    # Val acc (light dashed)
    handles.append(mlines.Line2D([], [], color=color, linestyle='--', lw=2, alpha=0.5,
                                 label=f"{prefix.replace('C2_', '')} val acc"))

fig3.legend(handles, [h.get_label() for h in handles],
            loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))


plt.subplots_adjust(bottom=0.2)

plt.savefig("results/CombiPlots/variants_dropout.png", dpi=400, bbox_inches="tight")
print("Saved C2/C6/C16 variants vs dropout figure..")



# Build figure 4 (3 rows, 1 col)
c2_variants = {
    "C2_1K_AdamW0001_SL_do3": "blue",
    "C2_lrScheduler_EarlyStopping": "orange",
}
c6_variants = {
    "C6_1K_AdamW0001_SL_do3": "blue",
    "C6_lrScheduler_EarlyStopping": "orange",
}
c16_variants = {
    "C16_1K_AdamW0001_SL_do3": "blue",
    "C16_lrScheduler_EarlyStopping": "orange",
}


fig4, (ax_c2, ax_c6, ax_c16) = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), constrained_layout=True)

# Per-row plotting (adjust y-lims to your scales; comment/remove if you prefer autoscale)
plot_task_row(ax_c2, c2_variants, "C2",
              xlim=(1, 1000),
              loss_ylim=(0.1, 1),   # your earlier note for C2-ish scale
              acc_ylim=(70, 102))          # set e.g. (0.85, 0.94) if your acc is 0–1; or (85,94) if in %
plot_task_row(ax_c6, c6_variants, "C6",
              xlim=(1, 1000),
              loss_ylim=(0.3, 1.6),
              acc_ylim=(70, 102))
plot_task_row(ax_c16, c16_variants, "C16",
              xlim=(1, 1000),
              loss_ylim=(0.4, 3.0),
              acc_ylim=(20, 102))

# One shared legend (use C2 keys for short labels)
# handles = [plt.Line2D([], [], color=col, lw=2) for col in c2_variants.values()]
# labels  = [k.replace("C2_", "") for k in c2_variants.keys()]
# fig3.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))

# Create custom legend entries
handles = []

# For each variant color
for prefix, color in c2_variants.items():
    # Train loss (dark solid)
    handles.append(mlines.Line2D([], [], color=color, linestyle='-', lw=2, alpha=0.9,
                                 label=f"{prefix.replace('C2_', '')} train loss"))
    # Val loss (light solid)
    handles.append(mlines.Line2D([], [], color=color, linestyle='-', lw=2, alpha=0.5,
                                 label=f"{prefix.replace('C2_', '')} val loss"))
    # Train acc (dark dashed)
    handles.append(mlines.Line2D([], [], color=color, linestyle='--', lw=2, alpha=0.9,
                                 label=f"{prefix.replace('C2_', '')} train acc"))
    # Val acc (light dashed)
    handles.append(mlines.Line2D([], [], color=color, linestyle='--', lw=2, alpha=0.5,
                                 label=f"{prefix.replace('C2_', '')} val acc"))

fig3.legend(handles, [h.get_label() for h in handles],
            loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
fig3.tight_layout(rect=[0, 0.05, 1, 1])  # leave 5% margin at bottom



plt.subplots_adjust(bottom=0.2)

plt.savefig("results/CombiPlots/lrScheduler_EarlyStopping.png", dpi=400, bbox_inches="tight")
print("Saved C2/C6/C16 variants vs lrScheduler_EarlyStopping figure..")



# Build figure 5 (3 rows, 1 col)
c2_variants = {
    "C2_AdamW0001_SmoothLabel_lr001_do1": "blue",
    "C2_1K_AdamW0001_SL_do3": "green",
    "C2_AdW0001_LS_Lr001_do6": "red",
}
c6_variants = {
    "C6_AdamW0001_SmoothLabel_lr001_do1": "blue",
    "C6_1K_AdamW0001_SL_do3": "green",
    "C6_AdW0001_LS_Lr001_do6": "red",
}
c16_variants = {
    "C16_AdamW0001_SmoothLabel_lr001_do1": "blue",
    "C16_1K_AdamW0001_SL_do3": "green",
    "C16_AdW0001_LS_Lr001_do6": "red",
}


fig5, (c2, c6, c16) = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), constrained_layout=True)

# Per-row plotting (adjust y-lims to your scales; comment/remove if you prefer autoscale)
plot_task_row(c2, c2_variants, "C2",
              xlim=(1, 200),
              loss_ylim=None,
              acc_ylim=None, plot_Train=False)
plot_task_row(c6, c6_variants, "C6",
              xlim=(1, 200),
              loss_ylim=None,
              acc_ylim=None, plot_Train=False)
plot_task_row(c16, c16_variants, "C16",
              xlim=(1, 200),
              loss_ylim=None,
              acc_ylim=None, plot_Train=False)



# Create custom legend entries
handles = []

# For each variant color
for prefix, color in c2_variants.items():
    # Train loss (dark solid)
    # handles.append(mlines.Line2D([], [], color=color, linestyle='-', lw=2, alpha=0.9,
    #                              label=f"{prefix.replace('C2_', '')} train loss"))
    # Val loss (light solid)
    handles.append(mlines.Line2D([], [], color=color, linestyle='-', lw=2, alpha=0.4,
                                 label=f"{prefix.replace('C2_', '')} val loss"))
    # Train acc (dark dashed)
    # handles.append(mlines.Line2D([], [], color=color, linestyle='--', lw=2, alpha=0.9,
    #                              label=f"{prefix.replace('C2_', '')} train acc"))
    # Val acc (light dashed)
    handles.append(mlines.Line2D([], [], color=color, linestyle='--', lw=2, alpha=0.4,
                                 label=f"{prefix.replace('C2_', '')} val acc"))

fig5.legend(handles, [h.get_label() for h in handles],
            loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
fig5.tight_layout(rect=[0, 0.05, 1, 1])  # leave 5% margin at bottom

plt.subplots_adjust(bottom=0.15)

plt.savefig("results/CombiPlots/dropout_comparisons.png", dpi=400, bbox_inches="tight")
print("Saved C2/C6/C16 variants vs dropout figure..")