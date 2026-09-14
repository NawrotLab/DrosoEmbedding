"""
Analysis: State-axis valence interaction test
=============================================
Tests whether stimulus valence systematically displaces centroids along the
state axis beyond orthogonal encoding.

Across 50 runs (State_Modality_Valence_16, dmodel=16):
  1. Load the 16 class centroids from each run's test-set embeddings (from disk).
  2. Construct state axis = (mean_fed - mean_starved) / norm per run.
  3. Project all 16 centroids onto the state axis.
  4. Group by valence (pure appetitive / pure aversive / conflict) × state (fed/starved).
  5. Compute delta = mean(appetitive) - mean(aversive) per run, for fed and starved.
  6. Bootstrap 95% CI across runs; permutation test against shuffled valence labels.

Saves:
  results/CombiPlots/state_valence_interaction_test.pdf
  results/CombiPlots/state_valence_interaction_stats.json
"""

import os
import gc
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.utils.helpers import load_all_results, get_style
from src.utils.config_loader import load_config
from src.visualization.figure_base import apply_style, FONT_SIZES

apply_style()

BASE_RESULTS_DIR = load_config()['paths']['eval_base_dir']
TASK     = 'State_Modality_Valence_16'
OUT_PDF  = 'results/CombiPlots/state_valence_interaction_test.pdf'
OUT_JSON = 'results/CombiPlots/state_valence_interaction_stats.json'

N_BOOTSTRAP = 10_000
N_PERMUTE   = 10_000
RNG_SEED    = 42

VALENCE_COLORS = {
    'appetitive': '#2a9d8f',
    'aversive':   '#e76f51',
    'conflict':   '#7b4278',
}


# ─── Class classification ────────────────────────────────────────────────────

def classify_class(name: str):
    """Return (state, valence) from a LaTeX-formatted class name."""
    state   = 'starved' if '(S)' in name else 'fed'
    has_pos = '$^{+}$' in name
    has_neg = '$^{-}$' in name
    if has_pos and not has_neg:
        valence = 'appetitive'
    elif has_neg and not has_pos:
        valence = 'aversive'
    else:
        valence = 'conflict'
    return state, valence


# ─── Per-run computation ─────────────────────────────────────────────────────

def compute_run_projections(X, labels, class_names):
    """Compute per-class centroid projections onto the state axis for one run."""
    unique   = np.sort(np.unique(labels))
    centroids = np.array([X[labels == l].mean(axis=0) for l in unique])

    fed_inds     = [i for i, n in enumerate(class_names) if '(F)' in n]
    starved_inds = [i for i, n in enumerate(class_names) if '(S)' in n]

    mean_fed     = centroids[fed_inds].mean(axis=0)
    mean_starved = centroids[starved_inds].mean(axis=0)
    state_axis   = mean_fed - mean_starved
    state_axis  /= np.linalg.norm(state_axis)

    return centroids @ state_axis  # shape (n_classes,)


def group_projections(projections, class_names):
    """Group scalar projections by (state, valence)."""
    groups = {}
    for i, name in enumerate(class_names):
        key = classify_class(name)
        groups.setdefault(key, []).append(projections[i])
    return {k: np.array(v) for k, v in groups.items()}


PAIRS = [
    ('appetitive', 'aversive'),
    ('appetitive', 'conflict'),
    ('aversive',   'conflict'),
]
PAIR_LABELS = {
    ('appetitive', 'aversive'): 'app−avr',
    ('appetitive', 'conflict'): 'app−con',
    ('aversive',   'conflict'): 'avr−con',
}


def compute_delta(groups, state, val_a='appetitive', val_b='aversive'):
    a = groups.get((state, val_a), np.array([]))
    b = groups.get((state, val_b), np.array([]))
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(a.mean() - b.mean())


# ─── Statistics ──────────────────────────────────────────────────────────────

def bootstrap_ci(values, n=N_BOOTSTRAP, seed=RNG_SEED):
    rng   = np.random.default_rng(seed)
    boots = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n)]
    return np.percentile(boots, [2.5, 97.5])


def permutation_p(observed_mean, all_run_projs, class_names, state,
                  val_a='appetitive', val_b='aversive',
                  n=N_PERMUTE, seed=RNG_SEED):
    """One-sided p-value: P(perm_delta >= observed) under shuffled valence labels."""
    rng        = np.random.default_rng(seed)
    state_inds = [i for i, nm in enumerate(class_names)
                  if classify_class(nm)[0] == state]
    perm_means = []
    for _ in range(n):
        run_deltas = []
        for projs in all_run_projs:
            shuffled             = projs.copy()
            shuffled[state_inds] = rng.permutation(shuffled[state_inds])
            g = group_projections(shuffled, class_names)
            run_deltas.append(compute_delta(g, state, val_a, val_b))
        perm_means.append(np.nanmean(run_deltas))
    return float(np.mean(np.array(perm_means) >= observed_mean))


# ─── Data loading ────────────────────────────────────────────────────────────

def load_run_projections(task_entry, class_names):
    """Load each H16 run pickle, compute projections, return list of arrays."""
    all_projs = []
    h16_runs  = task_entry.get('runs', {}).get('H16', [])
    print(f"H16 runs found: {len(h16_runs)}")
    for run in h16_runs:
        path = run.get('path')
        if path is None:
            continue
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            X      = data.get('transformer_latent_space')
            labels = data.get('latent_labels')
            del data
            gc.collect()
            if X is None or labels is None:
                continue
            all_projs.append(compute_run_projections(X, labels, class_names))
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
    return all_projs


# ─── Figure ──────────────────────────────────────────────────────────────────

def make_figure(all_groups, stats, out_path):
    rng      = np.random.default_rng(RNG_SEED)
    valences = ['appetitive', 'aversive', 'conflict']
    y_base       = {v: i * 0.55 for i, v in enumerate(valences)}
    state_offset = {'fed': 0.0, 'starved': 0.0}
    jitter       = 0.07

    STATE_STYLE = {
        'fed':     dict(marker='D', ls='-',  mfc_fn=lambda c: c,       mec_fn=lambda c: 'white'),
        'starved': dict(marker='o', ls='--', mfc_fn=lambda c: 'white', mec_fn=lambda c: c),
    }

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle('State-axis projection by valence group',
                 fontsize=FONT_SIZES['title'], weight='bold')

    for state, st in STATE_STYLE.items():
        # Jittered run points
        for run_groups in all_groups:
            for val in valences:
                vals = run_groups.get((state, val), np.array([]))
                if len(vals) == 0:
                    continue
                y    = y_base[val] + state_offset[state]
                y_jit = y + rng.uniform(-jitter, jitter)
                c    = VALENCE_COLORS[val]
                ax.scatter(vals.mean(), y_jit,
                           color=st['mfc_fn'](c), edgecolors=st['mec_fn'](c) if st['marker'] == 'o' else c,
                           alpha=0.45, s=25, zorder=3, linewidths=0.8)

        # Grand mean + bootstrap CI
        for val in valences:
            run_means = np.array([
                g.get((state, val), np.array([])) for g in all_groups
            ], dtype=object)
            run_means = np.array([m.mean() for m in run_means if len(m) > 0])
            if len(run_means) == 0:
                continue
            grand = run_means.mean()
            ci    = bootstrap_ci(run_means)
            y     = y_base[val] + state_offset[state]
            c     = VALENCE_COLORS[val]
            ax.plot([ci[0], ci[1]], [y, y], st['ls'],
                    color=c, lw=2.5, zorder=4)
            ax.plot(grand, y, st['marker'],
                    color=c, ms=8, zorder=5,
                    markerfacecolor=st['mfc_fn'](c),
                    markeredgecolor=st['mec_fn'](c) if st['marker'] == 'o' else 'white',
                    markeredgewidth=1.2)

    y_top = max(y_base.values()) + 0.85   # headroom for annotation box
    ax.set_ylim(min(y_base.values()) - 0.35, y_top)
    ax.axvline(0, color='gray', lw=0.8, alpha=0.5, linestyle='--')
    ax.set_xlim(-1.6, 1.6)
    ax.set_xlabel('State-axis projection', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('Valence group', fontsize=FONT_SIZES['label'])
    ax.set_yticks(list(y_base.values()))
    ax.set_yticklabels([v.capitalize() for v in valences],
                       fontsize=FONT_SIZES['tick'])
    for lbl, val in zip(ax.get_yticklabels(), valences):
        lbl.set_color(VALENCE_COLORS[val])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # State legend (fed / starved marker style)
    legend_handles = [
        Line2D([0], [0], marker='D', color='0.4', ls='-',  label='Fed',
               markerfacecolor='0.4', markeredgecolor='white', ms=8, lw=1.8),
        Line2D([0], [0], marker='o', color='0.4', ls='--', label='Starved',
               markerfacecolor='white', markeredgecolor='0.4', ms=8, lw=1.8),
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              fontsize=FONT_SIZES['legend'], frameon=False)

    # Single annotation box at top centre — all 6 pairwise deltas
    lines = []
    for state in ('fed', 'starved'):
        parts = [f"{lbl}={stats[state][lbl]['mean_delta']:+.2f}"
                 for lbl in PAIR_LABELS.values()]
        lines.append(f"{state.capitalize()}:  " + "   ".join(parts))
    ax.text(0.5, 0.97, "\n".join(lines),
            transform=ax.transAxes, ha='center', va='top',
            fontsize=FONT_SIZES['small'], color='0.3',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='0.8', alpha=0.85))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f"{os.path.splitext(out_path)[0]}.{ext}", bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_path} + .png")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    styles, TASK_CLASS_NAMES, *_ = get_style(style='styles')
    class_names = TASK_CLASS_NAMES[TASK]

    results_dict = load_all_results(BASE_RESULTS_DIR, TASK_CLASS_NAMES,
                                    fixed_trf_for_E=16, fixed_cnn_for_H=16,
                                    only_cnn_dim=16)

    all_run_projs = load_run_projections(results_dict[TASK], class_names)
    print(f"Runs with latent data: {len(all_run_projs)}")
    if not all_run_projs:
        raise ValueError("No runs with latent space data found in H16 group.")

    all_run_projs_np = np.array(all_run_projs)  # (n_runs, 16)

    # Per-run grouped projections
    all_groups = [group_projections(p, class_names) for p in all_run_projs]

    # Statistics — all 3 pairwise deltas × 2 states
    stats = {}
    for state in ('fed', 'starved'):
        stats[state] = {}
        print(f"\n{state.upper()}")
        for val_a, val_b in PAIRS:
            label = PAIR_LABELS[(val_a, val_b)]
            d  = np.array([compute_delta(g, state, val_a, val_b) for g in all_groups])
            ci = bootstrap_ci(d)
            p  = permutation_p(float(d.mean()), all_run_projs_np, class_names,
                               state, val_a, val_b)
            stats[state][label] = {
                'mean_delta':              float(d.mean()),
                'std_delta':               float(d.std()),
                'ci_95':                   [float(ci[0]), float(ci[1])],
                'p_permutation_one_sided': float(p),
                'n_runs':                  int(len(d)),
            }
            p_str = 'p < 0.001' if p < 0.001 else f'p = {p:.4f}'
            print(f"  {label}: {d.mean():.4f} ± {d.std():.4f}  CI [{ci[0]:.4f}, {ci[1]:.4f}]  {p_str}")

    make_figure(all_groups, stats, OUT_PDF)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats: {OUT_JSON}")


if __name__ == '__main__':
    main()
