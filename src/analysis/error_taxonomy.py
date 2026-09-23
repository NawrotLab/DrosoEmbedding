"""
Error classification taxonomy for the 6-class (State x Modality) and
16-class (State x Modality x Valence) confusion matrices.

Used by scripts/figures/run_fig3_accuracy_error.py (Fig 3).
"""

import numpy as np

# ── 6-class error taxonomy ──────────────────────────────────────────────────
# Index order matches TASK_CLASS_NAMES['State_Modality_6']:
#   "Odor (S)", "Odor (F)", "Taste (S)", "Taste (F)",
#   "Odor + Taste (S)", "Odor + Taste (F)"
CLASS_PROPS_6 = [
    ('Odor',     'Starved'),   # 0
    ('Odor',     'Fed'),       # 1
    ('Taste',    'Starved'),   # 2
    ('Taste',    'Fed'),       # 3
    ('Combined', 'Starved'),   # 4
    ('Combined', 'Fed'),       # 5
]

ERR6_KEYS    = ['State', 'Modality', 'State\n× Modality']
ERR6_COLOURS = ['#0072B2', '#009E73', '#D55E00']

# ── 16-class error taxonomy ─────────────────────────────────────────────────
CLASS_PROPS_16 = [
    ('Odor',     'Starved', +1),        # 0
    ('Odor',     'Starved', -1),        # 1
    ('Odor',     'Fed',     +1),        # 2
    ('Odor',     'Fed',     -1),        # 3
    ('Taste',    'Starved', +1),        # 4
    ('Taste',    'Starved', -1),        # 5
    ('Taste',    'Fed',     +1),        # 6
    ('Taste',    'Fed',     -1),        # 7
    ('Combined', 'Starved', (+1, +1)),  # 8
    ('Combined', 'Starved', (-1, -1)),  # 9
    ('Combined', 'Starved', (-1, +1)),  # 10  conflict
    ('Combined', 'Starved', (+1, -1)),  # 11  conflict
    ('Combined', 'Fed',     (+1, +1)),  # 12
    ('Combined', 'Fed',     (-1, -1)),  # 13
    ('Combined', 'Fed',     (-1, +1)),  # 14  conflict
    ('Combined', 'Fed',     (+1, -1)),  # 15  conflict
]

ERR16_TYPE_ORDER = [1, 2, 3, 4, 6, 5, 7]
ERR16_XLABELS = [
    'Valence', 'State', 'Modality',
    'Valence\n× State', 'Valence\n× Modality',
    'State\n× Modality', 'Valence\n× State\n× Modality',
]
ERR16_COLOURS = {
    1: '#E69F00',
    2: '#0072B2',
    3: '#009E73',
    4: '#CC79A7',
    5: '#D55E00',
    6: '#56B4E9',
    7: '#999999',
}

MODALITY_IDX_16 = {
    mod: [i for i in range(16) if CLASS_PROPS_16[i][0] == mod]
    for mod in ('Odor', 'Taste', 'Combined')
}

T7_MODALITY_PAIRS = [('Combined', 'Odor'), ('Combined', 'Taste'), ('Odor', 'Taste')]


# ── Error classification functions ──────────────────────────────────────────

def classify_error_6(i, j):
    mod_i, state_i = CLASS_PROPS_6[i]
    mod_j, state_j = CLASS_PROPS_6[j]
    if mod_i == mod_j:    return 'State'
    if state_i == state_j: return 'Modality'
    return 'State\n× Modality'


def error_type_pct_6(cm):
    counts = {k: 0 for k in ERR6_KEYS}
    total_pred = int(np.array(cm).sum())
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            n = int(cm[i, j])
            if n == 0:
                continue
            counts[classify_error_6(i, j)] += n
    if total_pred == 0:
        return {k: 0.0 for k in ERR6_KEYS}
    return {k: 100.0 * v / total_pred for k, v in counts.items()}


def net_valence_16(cls_idx):
    _, _, val = CLASS_PROPS_16[cls_idx]
    if isinstance(val, tuple):
        return val[0] if val[0] == val[1] else 0
    return val


def classify_error_16(i, j):
    mod_i, state_i, _ = CLASS_PROPS_16[i]
    mod_j, state_j, _ = CLASS_PROPS_16[j]
    same_mod   = (mod_i   == mod_j)
    same_state = (state_i == state_j)
    nv_i, nv_j = net_valence_16(i), net_valence_16(j)
    same_val   = (nv_i != 0 and nv_j != 0 and nv_i == nv_j)

    if same_mod:
        if same_state:                       return 1   # Valence only
        if same_val:                         return 2   # State only
        return 4                                        # State × Valence
    else:
        if same_state and same_val:          return 3   # Modality only
        if not same_state and same_val:      return 5   # State × Modality
        if same_state and not same_val:      return 6   # Modality × Valence
        return 7                                        # All three


def error_type_pct_16(cm):
    counts = {t: 0 for t in range(1, 8)}
    total_pred = int(np.array(cm).sum())
    for i in range(16):
        for j in range(16):
            if i == j:
                continue
            n = int(cm[i, j])
            if n == 0:
                continue
            counts[classify_error_16(i, j)] += n
    if total_pred == 0:
        return {t: 0.0 for t in range(1, 8)}
    return {t: 100.0 * v / total_pred for t, v in counts.items()}


# ── Extra 16-class diagnostics (manuscript claim checks) ────────────────────

def modality_error_breakdown_16(cm, true_idx):
    """For true labels restricted to `true_idx` (all same modality), split
    misclassifications into same-modality/diff-state, same-modality/same-state
    (valence-only), and different-modality-entirely."""
    same_mod_diff_state = same_mod_same_state = diff_mod = 0
    for i in true_idx:
        mod_i, state_i, _ = CLASS_PROPS_16[i]
        for j in range(16):
            if i == j:
                continue
            n = int(cm[i, j])
            if n == 0:
                continue
            mod_j, state_j, _ = CLASS_PROPS_16[j]
            if mod_j == mod_i:
                if state_j == state_i:
                    same_mod_same_state += n
                else:
                    same_mod_diff_state += n
            else:
                diff_mod += n
    total = same_mod_diff_state + same_mod_same_state + diff_mod
    if total == 0:
        return None
    return {
        'same_modality_diff_state': same_mod_diff_state / total,
        'same_modality_same_state_valence_only': same_mod_same_state / total,
        'different_modality': diff_mod / total,
    }


def t7_modality_involvement_16(cm):
    """For T7 (Valence×State×Modality) errors, fraction of error mass where
    true-or-predicted label falls in Odor / Taste / Combined, plus the
    pairwise modality split (T7 always spans exactly two modalities)."""
    counts = {'Odor': 0, 'Taste': 0, 'Combined': 0}
    pair_counts = {p: 0 for p in T7_MODALITY_PAIRS}
    total = 0
    for i in range(16):
        for j in range(16):
            if i == j or classify_error_16(i, j) != 7:
                continue
            n = int(cm[i, j])
            if n == 0:
                continue
            mod_i, _, _ = CLASS_PROPS_16[i]
            mod_j, _, _ = CLASS_PROPS_16[j]
            total += n
            counts[mod_i] += n
            counts[mod_j] += n
            pair_counts[tuple(sorted((mod_i, mod_j)))] += n
    if total == 0:
        return None
    return {
        'frac_odor':     counts['Odor'] / total,
        'frac_taste':    counts['Taste'] / total,
        'frac_combined': counts['Combined'] / total,
        'pair_fracs':    {p: c / total for p, c in pair_counts.items()},
    }


def error_type_pct_16_subset(cm, true_idx):
    """Same 7-category decomposition as error_type_pct_16, but restricted to
    rows (true labels) in `true_idx`; denominator is errors from that subset only."""
    counts = {t: 0 for t in range(1, 8)}
    total = 0
    for i in true_idx:
        for j in range(16):
            if i == j:
                continue
            n = int(cm[i, j])
            if n == 0:
                continue
            counts[classify_error_16(i, j)] += n
            total += n
    if total == 0:
        return None
    return {t: 100.0 * v / total for t, v in counts.items()}
