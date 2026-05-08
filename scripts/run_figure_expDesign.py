"""
Figure 1 — Experimental Setup and Example Recordings
=====================================================
Panel A : LFM microscopy setup  (SVG → cairosvg → imshow)   ┐
Panel B : Experimental design timeline                       ┘ narrow left column
Panel C : 8 grouping rows × (6 included + 1 excluded)         wide right column

Each column within a row comes from a **different** recording for diversity.
All frames are resized to a uniform shape (min H × min W) and normalised
globally (1st / 99th percentile across the entire panel).
"""

import os, re, io, random

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import tifffile
import cairosvg

from src.visualization.figure_base import apply_style, FONT_SIZES, PAGE_WIDTH

# ─── FONT ────────────────────────────────────────────────────────────────────
apply_style()
mpl.rcParams.update({
    'figure.dpi'  : 300,
    'svg.fonttype': 'none',
})

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SEED        = 1111
PANEL_A_SVG = "/rhomes/aabdel/DrosoEmbedding/src/src_imgs/LFM_Sketch.svg"

DATA_ROOT  = "/projects/lab-data/Collaboration/Gruenwald_Kadow/imgs4DL"
ALL_TS_DIR = os.path.join(DATA_ROOT, "meanZ_allTs")
LOG_TS_DIR = os.path.join(DATA_ROOT, "meanZ_logTs")

OUTPUT_BASE = "/rhomes/aabdel/DrosoEmbedding/results/CombiPlots/fig_expDesign"

N_INC = 5   # included frames per row
N_EXC = 1   # excluded frames per row

INC_COL = '#52b788'
EXC_COL = '#e05d44'

# Timeline — 5 key events
TIMELINE_EVENTS = [
    (0,   'Start of\nRecording'),
    (30,  '1$^{st}$ Stimulus\nPresentation'),
    (60,  '2$^{nd}$ Stimulus\nPresentation'),
    (90,  '3$^{rd}$ Stimulus\nPresentation'),
    (120, 'End of\nRecording'),
]

# ─── 8 GROUPINGS ────────────────────────────────────────────────────────────
GROUPINGS = [
    ("Starved",          lambda p: p['state'] == 'S'),
    ("Fed",              lambda p: p['state'] == 'F'),
    ("Odor",             lambda p: p['stim'] == 'O'),
    ("Taste",            lambda p: p['stim'] == 'T'),
    ("Odor + Taste",     lambda p: p['stim'] in ('MM', 'MC')),
    ("Appetitive",       lambda p: p['valence'] == 'P'),
    ("Aversive",         lambda p: p['valence'] == 'N'),
    ("App. + Aversive",  lambda p: p['stim'] == 'MC'),
]


# ─── CONDITION PARSING ──────────────────────────────────────────────────────
def parse_benz(benz):
    state    = benz[0]
    modality = benz[1]
    if modality == 'M':
        stim    = 'MM' if benz[2] == 'M' else 'MC'
        valence = benz[3]
    else:
        stim    = modality
        valence = benz[2]
    return dict(state=state, stim=stim, valence=valence, benz=benz)


def get_all_recordings(all_ts_dir):
    """Parse every recording directory; return [(rec_name, parsed), …]."""
    parsed = []
    for d in sorted(os.listdir(all_ts_dir)):
        if not os.path.isdir(os.path.join(all_ts_dir, d)):
            continue
        m = re.match(r'^([A-Z]+)_(\d+)$', d)
        if m:
            p = parse_benz(m.group(1))
            if p['valence'] != 'C':        # skip controls
                parsed.append((d, p))
    return parsed


# ─── INCLUDED / EXCLUDED FRAME INDICES (cached) ─────────────────────────────
_ie_cache = {}

def get_included_excluded(rec_name):
    """Return (sorted included indices, sorted excluded indices)."""
    if rec_name in _ie_cache:
        return _ie_cache[rec_name]

    all_dir = os.path.join(ALL_TS_DIR, rec_name)
    log_dir = os.path.join(LOG_TS_DIR, rec_name)

    all_f = {int(f.replace('.tiff', '').split('_')[-1])
             for f in os.listdir(all_dir) if f.endswith('.tiff')}
    log_f = {int(f.replace('.tiff', '').split('_')[-1])
             for f in os.listdir(log_dir) if f.endswith('.tiff')
             } if os.path.isdir(log_dir) else set()

    result = (sorted(log_f), sorted(all_f - log_f))
    _ie_cache[rec_name] = result
    return result


# ─── PER-COLUMN RECORDING DIVERSITY ─────────────────────────────────────────
def select_frames_for_panel(all_recs, groupings, seed, n_inc=6, n_exc=1):
    """
    For each grouping row, pick n_inc + n_exc **different** recordings.
    Each included slot → 1 random included frame from a distinct recording.
    Each excluded slot → 1 random excluded frame from yet another recording.

    Returns:
        [(label, inc_list, exc_list), …]
        inc_list / exc_list = [(rec_name, frame_idx), …]
    """
    rng = random.Random(seed)

    rows = []
    for label, filt in groupings:
        pool = [(r, p) for r, p in all_recs if filt(p)]
        rng.shuffle(pool)

        used_in_row = set()
        inc_entries = []
        exc_entries = []

        # ── Pick n_inc distinct recordings, each contributing 1 included frame
        for rec_name, _ in pool:
            if len(inc_entries) >= n_inc:
                break
            if rec_name in used_in_row:
                continue
            inc_idx, _ = get_included_excluded(rec_name)
            if inc_idx:
                inc_entries.append((rec_name, rng.choice(inc_idx)))
                used_in_row.add(rec_name)

        # ── Pick n_exc distinct recordings for excluded frames
        for rec_name, _ in pool:
            if len(exc_entries) >= n_exc:
                break
            if rec_name in used_in_row:
                continue
            _, exc_idx = get_included_excluded(rec_name)
            if exc_idx:
                exc_entries.append((rec_name, rng.choice(exc_idx)))
                used_in_row.add(rec_name)

        # Fallback: allow reuse if pool was too small
        if len(inc_entries) < n_inc:
            for rec_name, _ in pool:
                if len(inc_entries) >= n_inc:
                    break
                inc_idx, _ = get_included_excluded(rec_name)
                if inc_idx:
                    inc_entries.append((rec_name, rng.choice(inc_idx)))
            print(f"  NOTE: '{label}' reused recordings to fill "
                  f"{n_inc} included slots (pool size={len(pool)})")

        if len(exc_entries) < n_exc:
            for rec_name, _ in pool:
                if len(exc_entries) >= n_exc:
                    break
                _, exc_idx = get_included_excluded(rec_name)
                if exc_idx:
                    exc_entries.append((rec_name, rng.choice(exc_idx)))
            print(f"  NOTE: '{label}' reused recordings to fill "
                  f"{n_exc} excluded slots")

        rows.append((label, inc_entries, exc_entries))

    return rows


# ─── IMAGE UTILS ────────────────────────────────────────────────────────────
def load_svg(path):
    png_bytes = cairosvg.svg2png(url=path)
    return np.array(Image.open(io.BytesIO(png_bytes)))


def load_frame(rec_dir, frame_idx):
    rec_name  = os.path.basename(rec_dir)
    tiff_path = os.path.join(rec_dir, f"{rec_name}_{frame_idx}.tiff")
    if not os.path.exists(tiff_path):
        available = sorted(
            int(f.replace('.tiff', '').split('_')[-1])
            for f in os.listdir(rec_dir) if f.endswith('.tiff')
        )
        if not available:
            return np.zeros((50, 80))
        frame_idx = min(available, key=lambda x: abs(x - frame_idx))
        tiff_path = os.path.join(rec_dir, f"{rec_name}_{frame_idx}.tiff")
    return tifffile.imread(tiff_path).astype(float)


def resize_frame(arr, target_hw):
    """Resize 2D float array to (H, W) via bilinear interpolation.
    Operates on raw intensities BEFORE normalisation."""
    if arr.shape[:2] == target_hw:
        return arr
    img = Image.fromarray(arr)
    img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    return np.array(img, dtype=float)


def norm_frame(arr, lo, hi):
    """Normalise to [0, 1] using pre-computed lo/hi."""
    if hi == lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def get_recording_norm(rec_name, n_sample=200):
    """Per-recording 1st/99th percentile across ~n_sample evenly-spaced frames.
    This calibrates each brain to its own dynamic range so every frame
    in the panel gets good contrast regardless of absolute fluorescence."""
    rec_dir = os.path.join(ALL_TS_DIR, rec_name)
    files = sorted(f for f in os.listdir(rec_dir) if f.endswith('.tiff'))
    if not files:
        return 0.0, 1.0
    step = max(1, len(files) // n_sample)
    vals = np.concatenate([
        tifffile.imread(os.path.join(rec_dir, f)).astype(float).ravel()
        for f in files[::step]
    ])
    return float(np.nanpercentile(vals, 1)), float(np.nanpercentile(vals, 99))


# ─── TIMELINE ───────────────────────────────────────────────────────────────
def draw_timeline(ax):
    """Clean x-axis timeline inside its own subplot."""
    n = len(TIMELINE_EVENTS)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-1, 1)

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    ax.plot([0, n - 1], [0, 0], 'k-', linewidth=0.5, clip_on=False)

    for ci, (t_s, lbl) in enumerate(TIMELINE_EVENTS):
        ha = 'left' if ci == 0 else ('right' if ci == n - 1 else 'center')
        ax.plot(ci, 0, 'o', color='#1d3557', ms=8, zorder=5,
                markeredgecolor='white', markeredgewidth=1.0, clip_on=False)
        ax.text(ci, -0.35, f'{t_s} s', ha=ha, va='top', fontsize=FONT_SIZES['small'],
                style='italic', fontweight='bold', color='#1d3557',
                clip_on=False)
        ax.text(ci, 0.35, lbl, ha=ha, va='bottom', fontsize=FONT_SIZES['small'],
                color='#333333', linespacing=1.3, clip_on=False)


# ─── MAIN ───────────────────────────────────────────────────────────────────
def build_figure():
    print(f"\n{'=' * 60}")
    print(f"  SEED = {SEED}")
    print(f"{'=' * 60}")

    # ── Select per-column diverse recordings ──────────────────────────────
    all_recs = get_all_recordings(ALL_TS_DIR)
    print(f"Total non-control recordings found: {len(all_recs)}")

    row_data = select_frames_for_panel(all_recs, GROUPINGS, SEED,
                                       n_inc=N_INC, n_exc=N_EXC)

    print(f"\nPanel C recording assignments (each column = different recording):")
    for label, inc_entries, exc_entries in row_data:
        inc_info = [f"{r.split('_')[-1]}" for r, _ in inc_entries]
        exc_info = [f"{r.split('_')[-1]}" for r, _ in exc_entries]
        print(f"  {label:<18s}  inc recs: [{', '.join(inc_info):>30s}]"
              f"  exc recs: [{', '.join(exc_info)}]")

    # ── Collect all (rec_dir, frame_idx) specs and load frames ────────────
    all_specs = []
    for _, inc_entries, exc_entries in row_data:
        for rec_name, fr_idx in inc_entries:
            all_specs.append((os.path.join(ALL_TS_DIR, rec_name), fr_idx))
        for rec_name, fr_idx in exc_entries:
            all_specs.append((os.path.join(ALL_TS_DIR, rec_name), fr_idx))

    print(f"\nLoading {len(all_specs)} frames …")
    raw_frames = {}
    shapes = set()
    for rec_dir, fr_idx in all_specs:
        key = (rec_dir, fr_idx)
        if key not in raw_frames:
            arr = load_frame(rec_dir, fr_idx)
            raw_frames[key] = arr
            shapes.add(arr.shape[:2])

    # ── Uniform shape: downscale to min(H) × min(W) ──────────────────────
    if len(shapes) == 1:
        target_hw = shapes.pop()
        print(f"  All frames already {target_hw[0]}×{target_hw[1]}")
    else:
        min_h = min(s[0] for s in shapes)
        min_w = min(s[1] for s in shapes)
        target_hw = (min_h, min_w)
        print(f"  Found shapes: {sorted(shapes)}")
        print(f"  Resizing all to {target_hw[0]}×{target_hw[1]} (smallest)")

    for key in raw_frames:
        raw_frames[key] = resize_frame(raw_frames[key], target_hw)

    # ── Per-recording normalisation ──────────────────────────────────────
    # Each brain is calibrated to its own dynamic range (1st/99th pctl
    # across ~80 evenly-spaced frames from the full timeseries).
    # This avoids bright recordings washing out quiet ones.
    unique_recs = set()
    for _, inc_entries, exc_entries in row_data:
        for rec_name, _ in inc_entries:
            unique_recs.add(rec_name)
        for rec_name, _ in exc_entries:
            unique_recs.add(rec_name)

    print(f"\nPer-recording normalisation ({len(unique_recs)} unique recordings):")
    rec_norm = {}   # rec_name → (lo, hi)
    for rec_name in sorted(unique_recs):
        lo, hi = get_recording_norm(rec_name)
        rec_norm[rec_name] = (lo, hi)
        print(f"  {rec_name:<20s}  [{lo:.1f}, {hi:.1f}]")

    # ── Layout ────────────────────────────────────────────────────────────
    n_rows = len(GROUPINGS)
    fig = plt.figure(figsize=(18, 9))

    outer = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[0.28, 0.72],
        left=0.01, right=0.99,
        top=0.97, bottom=0.02,
        wspace=0.04,
    )

    # Left column: A (tall) over B (short)
    left_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0],
        height_ratios=[3.5, 1.0],
        hspace=0.12,
    )

    # Right column: C grid — 8 rows × (6 inc + gap + 1 exc)
    col_ratios = [1.0] * N_INC + [0.15] + [1.0] * N_EXC
    n_grid_cols = N_INC + 1 + N_EXC

    c_gs = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_grid_cols,
        subplot_spec=outer[1],
        width_ratios=col_ratios,
        hspace=0.08, wspace=0.04,
    )

    # ── Panel A ───────────────────────────────────────────────────────────
    ax_a = fig.add_subplot(left_gs[0])
    try:
        ax_a.imshow(load_svg(PANEL_A_SVG), aspect='equal')
    except Exception as e:
        ax_a.text(0.5, 0.5, f'[Panel A]\n{e}',
                  ha='center', va='center', transform=ax_a.transAxes,
                  fontsize=FONT_SIZES['annotation'], color='grey')
    ax_a.axis('off')

    # ── Panel B ───────────────────────────────────────────────────────────
    ax_b = fig.add_subplot(left_gs[1])
    draw_timeline(ax_b)

    # ── Panel C ───────────────────────────────────────────────────────────
    for ri, (label, inc_entries, exc_entries) in enumerate(row_data):

        # --- Included columns 0 … N_INC-1 ---
        for ci, (rec_name, fr_idx) in enumerate(inc_entries):
            rec_dir = os.path.join(ALL_TS_DIR, rec_name)
            frame = raw_frames[(rec_dir, fr_idx)]
            r_lo, r_hi = rec_norm[rec_name]

            ax = fig.add_subplot(c_gs[ri, ci])
            ax.imshow(norm_frame(frame, r_lo, r_hi),
                      cmap='viridis', aspect='equal', interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(INC_COL); sp.set_linewidth(1.5)

            if ci == 0:
                ax.set_ylabel(label.lower(), fontsize=FONT_SIZES['small'],
                              rotation=90, ha='center', va='center',
                              labelpad=8)

        # --- Gap column N_INC: no subplot ---

        # --- Excluded column(s) ---
        for ci, (rec_name, fr_idx) in enumerate(exc_entries):
            rec_dir = os.path.join(ALL_TS_DIR, rec_name)
            frame = raw_frames[(rec_dir, fr_idx)]
            r_lo, r_hi = rec_norm[rec_name]

            ax = fig.add_subplot(c_gs[ri, N_INC + 1 + ci])
            ax.imshow(norm_frame(frame, r_lo, r_hi),
                      cmap='viridis', aspect='equal', interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(EXC_COL); sp.set_linewidth(1.5)

    # ── Column headers ────────────────────────────────────────────────────
    c_bbox = c_gs.get_grid_positions(fig)
    inc_left  = c_bbox[2][0]
    inc_right = c_bbox[3][N_INC - 1]
    exc_left  = c_bbox[2][N_INC + 1]
    exc_right = c_bbox[3][N_INC + 1 + N_EXC - 1]
    top_y     = c_bbox[1][0] + 0.015

    fig.text((inc_left + inc_right) / 2, top_y, 'Included',
             ha='center', va='bottom', fontsize=FONT_SIZES['subplot_title'], fontweight='bold',
             color=INC_COL)
    fig.text((exc_left + exc_right) / 2, top_y, 'Excluded',
             ha='center', va='bottom', fontsize=FONT_SIZES['subplot_title'], fontweight='bold',
             color=EXC_COL)

    # ── Panel labels ──────────────────────────────────────────────────────
    lkw = dict(fontsize=FONT_SIZES['panel_label'], fontweight='bold', transform=fig.transFigure,
               va='top', ha='left')
    fig.text(0.01,  0.97, 'A', **lkw)
    fig.text(0.01,  0.25, 'B', **lkw)
    fig.text(0.295, 0.97, 'C', **lkw)

    # ── Save ──────────────────────────────────────────────────────────────
    out = f'{OUTPUT_BASE}_S{SEED}'
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(f'{out}.{ext}', bbox_inches='tight', dpi=300)
        print(f"Saved → {out}.{ext}")
    plt.close(fig)
    print("Done.")


if __name__ == '__main__':
    build_figure()