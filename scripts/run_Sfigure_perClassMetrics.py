"""
Supplementary Figure — Per-class F1, Precision, Recall (16-class task)
==========================================================================

Three groups on the x-axis: Odor (classes 0-3) | Taste (classes 4-7) | Combined (classes 8-15).
Within each group, three bars (one per metric) are coloured by group.
Individual class dots are overlaid; the lowest-recall class is marked with a small arrow.
A vertical legend on the right shows all 16 class symbols (Fed / Starved).

Usage (from repo root):
    python scripts/run_Sfigure_perClassMetrics.py

Output:
    results/CombiPlots/figS_perClassMetrics.pdf
    results/CombiPlots/figS_perClassMetrics.png
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization.figure_base import apply_style, FONT_SIZES
from src.visualization.visualize_performance import _plot_class_symbol
from src.utils.helpers import load_all_results

apply_style()

# ════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════

BASE_RESULTS_DIR = os.path.join('results', '_chkpt_finals')
OUT_DIR = os.path.join('results', 'CombiPlots')
TASK_KEY = 'State_Modality_Valence_16'

CLASS_NAMES_16 = [
    r'O$^{+}$ (S)',          r'O$^{-}$ (S)',          r'O$^{+}$ (F)',          r'O$^{-}$ (F)',
    r'T$^{+}$ (S)',          r'T$^{-}$ (S)',          r'T$^{+}$ (F)',          r'T$^{-}$ (F)',
    r'O$^{+}$+T$^{+}$ (S)', r'O$^{-}$+T$^{-}$ (S)',
    r'O$^{-}$+T$^{+}$ (S)', r'O$^{+}$+T$^{-}$ (S)',
    r'O$^{+}$+T$^{+}$ (F)', r'O$^{-}$+T$^{-}$ (F)',
    r'O$^{-}$+T$^{+}$ (F)', r'O$^{+}$+T$^{-}$ (F)',
]

STYLE_KEYS_16 = [
    'starved_odor_positive',      'starved_odor_negative',      'fed_odor_positive',      'fed_odor_negative',
    'starved_taste_positive',     'starved_taste_negative',     'fed_taste_positive',     'fed_taste_negative',
    'starved_odor_pos_taste_pos', 'starved_odor_neg_taste_neg',
    'starved_odor_neg_taste_pos', 'starved_odor_pos_taste_neg',
    'fed_odor_pos_taste_pos',     'fed_odor_neg_taste_neg',
    'fed_odor_neg_taste_pos',     'fed_odor_pos_taste_neg',
]

GROUPS = {
    'Odor':     list(range(0, 4)),
    'Taste':    list(range(4, 8)),
    'Combined': list(range(8, 16)),
}
GROUP_LABELS = ['Odor', 'Taste', 'Combined']
GROUP_COLOURS = {
    'Odor':     '#D35F2A',
    'Taste':    '#3382BE',
    'Combined': '#7B4278',
}

METRICS = ['f1-score', 'precision', 'recall']
METRIC_DISPLAY = {'f1-score': 'F1', 'precision': 'Prec', 'recall': 'Rec'}

LOWEST_CLASS  = r'T$^{+}$ (S)'
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
# COMPUTE GROUP AVERAGES AND PER-CLASS VALUES (%)
# ════════════════════════════════════════════════

per_class_vals = {}
group_avgs     = {}

for group_name, indices in GROUPS.items():
    per_class_vals[group_name] = {}
    group_avgs[group_name]     = {}
    for metric in METRICS:
        vals = [report[CLASS_NAMES_16[i]][metric] * 100 for i in indices]
        per_class_vals[group_name][metric] = vals
        group_avgs[group_name][metric]     = float(np.mean(vals))

# ════════════════════════════════════════════════
# FIGURE GEOMETRY
# ════════════════════════════════════════════════

BAR_W  = 0.18
GAP    = 0.30

n_metrics    = len(METRICS)
group_span   = n_metrics * BAR_W
group_step   = group_span + GAP
group_centers = {g: i * group_step for i, g in enumerate(GROUP_LABELS)}

metric_offsets = {
    m: (j - (n_metrics - 1) / 2) * BAR_W
    for j, m in enumerate(METRICS)
}

# ════════════════════════════════════════════════
# VERTICAL LEGEND (right panel)
# ════════════════════════════════════════════════

def _draw_vertical_legend(ax, all_styles):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ms = 9
    fs = FONT_SIZES['small']

    x_label = 0.02
    x_fed   = 0.72
    x_stv   = 0.90

    legend_groups = [
        ('Odor', GROUP_COLOURS['Odor'], [
            ('O app',       'fed_odor_positive',      'starved_odor_positive'),
            ('O avr',       'fed_odor_negative',       'starved_odor_negative'),
        ]),
        ('Taste', GROUP_COLOURS['Taste'], [
            ('T app',       'fed_taste_positive',      'starved_taste_positive'),
            ('T avr',       'fed_taste_negative',       'starved_taste_negative'),
        ]),
        ('O+T', GROUP_COLOURS['Combined'], [
            ('OT app',         'fed_odor_pos_taste_pos',  'starved_odor_pos_taste_pos'),
            ('OT avr',         'fed_odor_neg_taste_neg',  'starved_odor_neg_taste_neg'),
            (r'T$^+$O$^-$',    'fed_odor_neg_taste_pos',  'starved_odor_neg_taste_pos'),
            (r'T$^-$O$^+$',    'fed_odor_pos_taste_neg',  'starved_odor_pos_taste_neg'),
        ]),
    ]

    # Column headers
    y = 0.97
    ax.text(x_fed, y, 'Fed', ha='center', va='center', fontsize=fs, fontweight='bold', color='0.3')
    ax.text(x_stv, y, 'Stv', ha='center', va='center', fontsize=fs, fontweight='bold', color='0.3')

    row_h_group = 0.07
    row_h_item  = 0.065

    y = 0.90
    for group_name, group_color, items in legend_groups:
        ax.text(x_label, y, group_name,
                ha='left', va='center', fontsize=fs, fontweight='bold', color=group_color)
        y -= row_h_group

        for label, fed_key, stv_key in items:
            ax.text(x_label + 0.04, y, label,
                    ha='left', va='center', fontsize=fs, color='0.3')
            fed_s = all_styles.get(fed_key, {})
            stv_s = all_styles.get(stv_key, {})
            _plot_class_symbol(ax, x_fed, y, fed_s, markersize=ms, zorder=5)
            _plot_class_symbol(ax, x_stv, y, stv_s, markersize=ms, zorder=5)
            y -= row_h_item

        y -= 0.01  # extra gap between groups

    y_foot = max(0.04, y + 0.01)
    ax.text(0.5, y_foot, u'● Fed   ○ Stv',
            ha='center', va='center', fontsize=fs - 1, color='0.5')

# ════════════════════════════════════════════════
# PLOT
# ════════════════════════════════════════════════

fig = plt.figure(figsize=(10, 5))
gs  = GridSpec(1, 2, figure=fig,
               width_ratios=[3.8, 1],
               left=0.08, right=0.99,
               bottom=0.22, top=0.96,
               wspace=0.05)

ax     = fig.add_subplot(gs[0])
ax_leg = fig.add_subplot(gs[1])

rng = np.random.default_rng(42)
lowest_xy = (None, None)

# ── Bars coloured by group ──
for group_name in GROUP_LABELS:
    for metric in METRICS:
        x = group_centers[group_name] + metric_offsets[metric]
        y = group_avgs[group_name][metric]
        ax.bar(
            x, y,
            width=BAR_W * 0.88,
            color=GROUP_COLOURS[group_name],
            alpha=0.55,
            zorder=2,
        )

# ── Per-class dots ──
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

# ── Vertical metric labels below each bar ──
trans = blended_transform_factory(ax.transData, ax.transAxes)
for group_name in GROUP_LABELS:
    for metric in METRICS:
        x   = group_centers[group_name] + metric_offsets[metric]
        val = group_avgs[group_name][metric]
        lbl = f'{METRIC_DISPLAY[metric]} = {val:.0f}'
        ax.text(
            x, -0.04, lbl,
            ha='center', va='top',
            rotation=90,
            fontsize=FONT_SIZES['small'],
            color=GROUP_COLOURS[group_name],
            transform=trans,
            clip_on=False,
        )

# ── Group labels below metric text ──
for group_name in GROUP_LABELS:
    ax.text(
        group_centers[group_name], -0.22, group_name,
        ha='center', va='top',
        fontsize=FONT_SIZES['label'],
        color=GROUP_COLOURS[group_name],
        fontweight='bold',
        transform=trans,
        clip_on=False,
    )

# ── Vertical separators between groups ──
for a_grp, b_grp in [('Odor', 'Taste'), ('Taste', 'Combined')]:
    sep = (group_centers[a_grp] + group_centers[b_grp]) / 2
    ax.axvline(sep, color='#CCCCCC', linewidth=1, zorder=1)

# ── Axes limits and ticks ──
x_margin = BAR_W * 2
ax.set_xlim(
    group_centers['Odor']     - group_span / 2 - x_margin,
    group_centers['Combined'] + group_span / 2 + x_margin,
)
ax.set_ylim(50, 105)
ax.set_yticks(np.arange(50, 101, 10))
ax.set_xticks([group_centers[g] for g in GROUP_LABELS])
ax.set_xticklabels([''] * len(GROUP_LABELS))
ax.tick_params(axis='x', length=0)
ax.set_ylabel('Score (%)', fontsize=FONT_SIZES['label'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Small black arrow pointing to lowest performer ──
lx, ly = lowest_xy
if lx is not None:
    ax.annotate(
        '',
        xy=(lx, ly),
        xytext=(lx + 0.13, ly - 5),
        arrowprops=dict(arrowstyle='->', color='black', lw=0.8, mutation_scale=8),
    )

# ── Right vertical legend ──
_draw_vertical_legend(ax_leg, all_styles)

# ════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════

os.makedirs(OUT_DIR, exist_ok=True)
stem = os.path.join(OUT_DIR, 'figS_perClassMetrics')
fig.savefig(stem + '.pdf', dpi=300, bbox_inches='tight')
fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {stem}.pdf / .png')
