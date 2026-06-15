"""
Diagnostic: KO contrast comparison across fill variants
========================================================
Loads the group-level ΔAccuracy CSVs produced by diag_KO_permutation for
each fill variant and generates the same 3-panel contrast plot (State,
Modality, Valence) as in Fig 4 — one row per variant — so the approaches
can be compared directly.

Usage (from repo root):
    python -m scripts.diagnostics.diag_KO_contrast_comparison

Outputs:
    results/diagnostics/KO_comparison/KO_contrast_comparison.pdf / .png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.visualization.visualize_interpretability import (
    plot_contrasts_horizontal,
    DEFAULT_CONTRAST_SPEC,
)
from src.visualization.figure_base import apply_style, FONT_SIZES, save_figure

apply_style()

# ── config ────────────────────────────────────────────────────────────────────

# variant label → directory containing KO_delta_accuracy_groups.csv
VARIANTS = {
    'Zero-fill':  'results/diagnostics/KO_permutation',
    'Noise-fill': 'results/diagnostics/KO_permutation_noisefill',
    'Static':     'results/diagnostics/KO_permutation_static',
    'Shuffled':   'results/diagnostics/KO_permutation_shuffled',
}

# contrasts to compute from the 8 group rows
CONTRAST_PAIRS = {
    'Starved_minus_Fed':         ('Starved',     'Fed'),
    'Odor_minus_Taste':          ('Odor',        'Taste'),
    'Appetitive_minus_Aversive': ('Appetitive',  'Aversive'),
}

OUT_DIR = 'results/diagnostics/KO_comparison'


# ── helpers ───────────────────────────────────────────────────────────────────

def load_contrasts(diag_dir: str) -> pd.DataFrame | None:
    """
    Load KO_delta_accuracy_groups.csv and compute contrast rows.
    Returns DataFrame with rows = contrast names, columns = neuropils,
    or None if the file doesn't exist.
    """
    csv_path = os.path.join(diag_dir, 'KO_delta_accuracy_groups.csv')
    if not os.path.exists(csv_path):
        print(f'  [skip] not found: {csv_path}')
        return None

    groups_df = pd.read_csv(csv_path, index_col=0)   # rows = group names
    rows = {}
    for contrast_name, (g1, g2) in CONTRAST_PAIRS.items():
        if g1 in groups_df.index and g2 in groups_df.index:
            rows[contrast_name] = groups_df.loc[g1] - groups_df.loc[g2]
        else:
            print(f'  [warn] group missing for {contrast_name}: {g1} or {g2}')

    if not rows:
        return None
    return pd.DataFrame(rows).T   # rows = contrasts, cols = neuropils


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # load all available variants
    data = {}
    for name, diag_dir in VARIANTS.items():
        print(f'Loading {name} from {diag_dir} ...')
        df = load_contrasts(diag_dir)
        if df is not None:
            data[name] = df

    if not data:
        raise RuntimeError('No variant CSVs found. Run diag_KO_permutation first.')

    print(f'Loaded {len(data)} variants: {list(data.keys())}')

    # shared x-limit across all variants for fair visual comparison
    all_vals = np.concatenate([df.values.ravel() for df in data.values()])
    global_xlim = float(np.nanmax(np.abs(all_vals))) * 1.15

    n_variants = len(data)
    fig = plt.figure(figsize=(11, n_variants * 2.8))
    gs  = GridSpec(n_variants, 3, figure=fig,
                   hspace=0.55, wspace=0.08,
                   left=0.14, right=0.97, top=0.93, bottom=0.04)

    for row_idx, (variant_name, contrasts_df) in enumerate(data.items()):
        axes = [fig.add_subplot(gs[row_idx, col]) for col in range(3)]

        plot_contrasts_horizontal(
            axes=axes,
            group_contrasts_abs=contrasts_df,
            neuropil_names=list(contrasts_df.columns),
            sort_by_modality=True,
            uniform_xlim=False,   # we enforce shared scale below
            fontsize_title=FONT_SIZES.get('subplot_title', 9),
            fontsize_labels=FONT_SIZES.get('label', 7),
        )

        # enforce shared x-limit
        for ax in axes:
            ax.set_xlim(-global_xlim, global_xlim)

        # row label on the left
        axes[0].set_title(variant_name, loc='left',
                          fontsize=FONT_SIZES.get('label', 8),
                          fontweight='bold', pad=14, color='dimgrey')

    fig.suptitle('KO neuropil importance — contrast comparison across fill variants',
                 fontsize=FONT_SIZES.get('title', 10), y=0.98)

    out_path = os.path.join(OUT_DIR, 'KO_contrast_comparison')
    save_figure(fig, out_path + '.pdf', formats=('pdf', 'png'))
    print(f'Saved → {out_path}.pdf / .png')


if __name__ == '__main__':
    main()
