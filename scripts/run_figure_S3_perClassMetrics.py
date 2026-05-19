"""
Supplementary Figure S3 — Per-class F1, Precision, Recall (16-class task)
==========================================================================

Three groups on the x-axis: Odour (classes 0-3) | Taste (classes 4-7) | Combined (classes 8-15).
Within each group, three outline-only bars give the group-average F1, Precision, and Recall.
Individual class dots (per-class colours from styles.yaml) are overlaid on the corresponding bar.
The lowest performer (T+ starved taste appetitive, recall ≈0.65) is annotated with an arrow.

Usage (from repo root):
    python scripts/run_figure_S3_perClassMetrics.py

Output:
    /mnt/user-data/outputs/figS3_perClassMetrics.pdf
    /mnt/user-data/outputs/figS3_perClassMetrics.png
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Allow imports from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization.figure_base import apply_style, FONT_SIZES
from src.visualization.visualize_performance import _plot_class_symbol
from src.utils.helpers import load_all_results

apply_style()

# ════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════

BASE_RESULTS_DIR = os.path.join('results', '_chkpt_finals')
OUT_DIR = '/mnt/user-data/outputs'
TASK_KEY = 'State_Modality_Valence_16'

# 16 class names, in label order, matching classification_report_dict keys
CLASS_NAMES_16 = [
    r'O$^{+}$ (S)',         r'O$^{-}$ (S)',         r'O$^{+}$ (F)',         r'O$^{-}$ (F)',          # 0-3  Odour
    r'T$^{+}$ (S)',         r'T$^{-}$ (S)',         r'T$^{+}$ (F)',         r'T$^{-}$ (F)',          # 4-7  Taste
    r'O$^{+}$+T$^{+}$ (S)', r'O$^{-}$+T$^{-}$ (S)',                                                  # 8-9
    r'O$^{-}$+T$^{+}$ (S)', r'O$^{+}$+T$^{-}$ (S)',                                                  # 10-11
    r'O$^{+}$+T$^{+}$ (F)', r'O$^{-}$+T$^{-}$ (F)',                                                  # 12-13
    r'O$^{-}$+T$^{+}$ (F)', r'O$^{+}$+T$^{-}$ (F)',                                                  # 14-15 Combined
]

# styles.yaml keys aligned with CLASS_NAMES_16 (same index order)
STYLE_KEYS_16 = [
    'starved_odor_positive',      'starved_odor_negative',      'fed_odor_positive',      'fed_odor_negative',
    'starved_taste_positive',     'starved_taste_negative',     'fed_taste_positive',     'fed_taste_negative',
    'starved_odor_pos_taste_pos', 'starved_odor_neg_taste_neg',
    'starved_odor_neg_taste_pos', 'starved_odor_pos_taste_neg',
    'fed_odor_pos_taste_pos',     'fed_odor_neg_taste_neg',
    'fed_odor_neg_taste_pos',     'fed_odor_pos_taste_neg',
]

# Groups (indices into CLASS_NAMES_16 / STYLE_KEYS_16)
GROUPS = {
    'Odour':    list(range(0, 4)),
    'Taste':    list(range(4, 8)),
    'Combined': list(range(8, 16)),
}
GROUP_LABELS = ['Odour', 'Taste', 'Combined']
GROUP_COLOURS = {
    'Odour':    '#D35F2A',
    'Taste':    '#3382BE',
    'Combined': '#7B4278',
}

# Metrics — bar colours for the outline-only bars
METRICS = ['f1-score', 'precision', 'recall']
METRIC_DISPLAY = {'f1-score': 'F1', 'precision': 'Precision', 'recall': 'Recall'}
METRIC_COLOURS = {
    'f1-score':  '#333333',
    'precision': '#1F7AB8',
    'recall':    '#C05A14',
}

# Class to annotate as the lowest performer
LOWEST_CLASS  = r'T$^{+}$ (S)'   # starved taste appetitive
LOWEST_METRIC = 'recall'

# ════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════

print(f'Loading results from {BASE_RESULTS_DIR} …')
results_dict = load_all_results(BASE_RESULTS_DIR, task_names=[TASK_KEY])

if TASK_KEY not in results_dict or results_dict[TASK_KEY].get('best') is None:
    raise FileNotFoundError(
        f"No 'best' result found for '{TASK_KEY}' in {BASE_RESULTS_DIR}. "
        "Run run_evaluation.py first."
    )

report = results_dict[TASK_KEY]['best']['classification_report_dict']

missing = [cn for cn in CLASS_NAMES_16 if cn not in report]
if missing:
    raise KeyError(f"Class names absent from report dict: {missing}")

# ════════════════════════════════════════════════
# LOAD STYLES
# ════════════════════════════════════════════════

styles_path = os.path.join('src', 'visualization', 'styles.yaml')
with open(styles_path, 'r') as f:
    all_styles = yaml.safe_load(f)['styles']

class_styles = [all_styles.get(key, {}) for key in STYLE_KEYS_16]

# ════════════════════════════════════════════════
# COMPUTE GROUP AVERAGES AND PER-CLASS VALUES
# ════════════════════════════════════════════════

per_class_vals = {}   # [group][metric] = list of floats
group_avgs     = {}   # [group][metric] = float

for group_name, indices in GROUPS.items():
    per_class_vals[group_name] = {}
    group_avgs[group_name]     = {}
    for metric in METRICS:
        vals = [report[CLASS_NAMES_16[i]][metric] for i in indices]
        per_class_vals[group_name][metric] = vals
        group_avgs[group_name][metric]     = float(np.mean(vals))

# ════════════════════════════════════════════════
# FIGURE GEOMETRY
# ════════════════════════════════════════════════

BAR_W   = 0.18   # width of each bar
GAP     = 0.30   # extra gap between groups (on top of bar span)

# Group centres: space groups so there is a visible gap between them
n_metrics   = len(METRICS)
group_span  = n_metrics * BAR_W                  # total bar span per group
group_step  = group_span + GAP                   # centre-to-centre distance
group_centers = {
    g: i * group_step for i, g in enumerate(GROUP_LABELS)
}

# Metric offset within each group: evenly space around centre
metric_offsets = {
    m: (j - (n_metrics - 1) / 2) * BAR_W
    for j, m in enumerate(METRICS)
}

# ════════════════════════════════════════════════
# PLOT
# ════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 5))

rng = np.random.default_rng(42)

# Track lowest-performer dot position for annotation
lowest_xy = (None, None)

# ── Bars (group averages) ──
for group_name in GROUP_LABELS:
    for metric in METRICS:
        x = group_centers[group_name] + metric_offsets[metric]
        y = group_avgs[group_name][metric]
        ax.bar(
            x, y,
            width=BAR_W * 0.88,
            facecolor='none',
            edgecolor=METRIC_COLOURS[metric],
            linewidth=2,
            zorder=2,
        )

# ── Per-class dots overlaid on their metric bar ──
for group_name in GROUP_LABELS:
    indices = GROUPS[group_name]
    n_cls   = len(indices)
    for metric in METRICS:
        x_bar = group_centers[group_name] + metric_offsets[metric]
        vals  = per_class_vals[group_name][metric]
        jitter = rng.uniform(-BAR_W * 0.32, BAR_W * 0.32, size=n_cls)
        for k, (i_global, v) in enumerate(zip(indices, vals)):
            style = class_styles[i_global]
            x_dot = x_bar + jitter[k]
            _plot_class_symbol(ax, x_dot, v, style, markersize=6, zorder=5)
            if CLASS_NAMES_16[i_global] == LOWEST_CLASS and metric == LOWEST_METRIC:
                lowest_xy = (x_dot, v)

# ── Group-average F1 label above each F1 bar ──
for group_name in GROUP_LABELS:
    x = group_centers[group_name] + metric_offsets['f1-score']
    y = group_avgs[group_name]['f1-score']
    ax.text(
        x, y + 0.013, f'{y:.2f}',
        ha='center', va='bottom',
        fontsize=FONT_SIZES['small'],
        color=METRIC_COLOURS['f1-score'],
        fontweight='bold',
    )

# ── Annotate lowest performer ──
lx, ly = lowest_xy
if lx is not None:
    ax.annotate(
        LOWEST_CLASS + f'\nrecall ≈ {ly:.2f}',
        xy=(lx, ly),
        xytext=(lx + 0.22, ly - 0.10),
        fontsize=FONT_SIZES['small'],
        color='#3382BE',
        arrowprops=dict(arrowstyle='->', color='#3382BE', lw=1.2),
        ha='left', va='top',
    )

# ── Group x-axis labels (coloured) ──
for group_name in GROUP_LABELS:
    ax.text(
        group_centers[group_name], -0.055, group_name,
        ha='center', va='top',
        fontsize=FONT_SIZES['label'],
        color=GROUP_COLOURS[group_name],
        fontweight='bold',
        transform=ax.get_xaxis_transform(),
    )

# ── Vertical separators between groups ──
for a_grp, b_grp in [('Odour', 'Taste'), ('Taste', 'Combined')]:
    sep = (group_centers[a_grp] + group_centers[b_grp]) / 2
    ax.axvline(sep, color='#CCCCCC', linewidth=1, zorder=1)

# ── Axes limits and ticks ──
x_margin = BAR_W * 2
ax.set_xlim(
    group_centers['Odour']    - group_span / 2 - x_margin,
    group_centers['Combined'] + group_span / 2 + x_margin,
)
ax.set_ylim(0.50, 1.05)
ax.set_yticks(np.arange(0.5, 1.01, 0.1))
ax.set_xticks([group_centers[g] for g in GROUP_LABELS])
ax.set_xticklabels([''] * len(GROUP_LABELS))   # labels drawn manually above
ax.tick_params(axis='x', length=0)
ax.set_ylabel('Score', fontsize=FONT_SIZES['label'])
ax.set_title('Per-class performance — State, Modality, Valence (16 classes)',
             fontsize=FONT_SIZES['subplot_title'], pad=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Legend for metrics ──
legend_handles = [
    mpatches.Patch(
        facecolor='none', edgecolor=METRIC_COLOURS[m], linewidth=2,
        label=METRIC_DISPLAY[m],
    )
    for m in METRICS
]
ax.legend(
    handles=legend_handles,
    loc='lower right',
    fontsize=FONT_SIZES['legend'],
    frameon=False,
)

# ── Panel label ──
ax.text(
    -0.07, 1.02, 'S3',
    transform=ax.transAxes,
    fontsize=FONT_SIZES['panel_label'],
    fontweight='bold',
    va='bottom',
)

# ════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════

os.makedirs(OUT_DIR, exist_ok=True)
stem = os.path.join(OUT_DIR, 'figS3_perClassMetrics')
fig.savefig(stem + '.pdf', dpi=300, bbox_inches='tight')
fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {stem}.pdf / .png')
