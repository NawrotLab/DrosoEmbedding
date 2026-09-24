"""
Shared constants and utilities for all figure scripts.

Single source of truth for figure width, font sizes, margins, DPI, and saving.
"""

import os
import matplotlib.pyplot as plt


# ── Render dimensions ─────────────────────────────────────────────────────────
# All main figures share this width; journals scale the PDF to their column width.
FIGURE_WIDTH = 18.0   # inches

# ── Output ────────────────────────────────────────────────────────────────────
DPI = 300

# ── Standard GridSpec margins (figure fraction) ───────────────────────────────
# bottom=0.15 leaves room for the shared legend panel; override per figure if needed.
MARGINS = dict(left=0.06, right=0.96, top=0.93, bottom=0.15)

# ── Font sizes (calibrated for FIGURE_WIDTH) ──────────────────────────────────
FONT_SIZES = {
    'panel_label':   19,   # a, b, c … panel letters (bold)
    'title':         17,   # structural row/column headers
    'subplot_title': 12,   # individual subplot titles
    'label':         14,   # xlabel / ylabel
    'tick':          14,   # tick labels
    'legend':        12,   # legend text
    'annotation':    15,   # in-plot text, accuracy numbers
    'colorbar':      12,   # colorbar tick labels and title
    'heatmap_cell':  11,   # text inside heatmap cells
    'legend_panel':  13,   # legend panel headers and row labels
    'small':         10,   # dense labels in tight layouts
}

# ── Task metadata (used by figure scripts that need chance levels / baselines) ─
TASK_CONFIG = {
    'MetabolicState_2': {
        'name':         'i. State',
        'chance_level': 50.0,
        'baseline':     54.1,
        'y_range':      (70, 100),
    },
    'State_Modality_6': {
        'name':         'ii. State, Modality',
        'chance_level': 100 / 6,
        'baseline':     27.8,
        'y_range':      (70, 100),
    },
    'State_Modality_Valence_16': {
        'name':         'iii. State, Modality, Valence',
        'chance_level': 100 / 16,
        'baseline':     10.0,
        'y_range':      (60, 90),
    },
}


def apply_style():
    """Call once at the top of each figure script before any plotting."""
    plt.rcParams.update({
        # DejaVu Sans, not Arial: Arial isn't installed on the cluster or on a
        # typical reviewer's machine, so matplotlib silently substitutes
        # DejaVu Sans anyway -- declaring it directly renders identically
        # (confirmed: same font actually used either way) without the
        # "findfont: Font family 'Arial' not found" warning on every text
        # element (hundreds of lines per figure, pure log noise).
        'font.family':     'DejaVu Sans',
        'font.size':        FONT_SIZES['annotation'],
        'axes.titlesize':   FONT_SIZES['subplot_title'],
        'axes.labelsize':   FONT_SIZES['label'],
        'xtick.labelsize':  FONT_SIZES['tick'],
        'ytick.labelsize':  FONT_SIZES['tick'],
        'legend.fontsize':  FONT_SIZES['legend'],
        'figure.titlesize': FONT_SIZES['title'],
    })


def add_panel_label(fig, label, ax=None, x=None, y=None, dx=None, dy=None, va='top'):
    """Draw a bold panel-letter label ('a', 'b', ...) in the shared house style.

    Single source of truth for panel-label font/size/weight/format so every
    figure renders them identically. Two ways to position it:
      - pass `ax` to anchor the label just above that axes' top-left corner
        (dx/dy are axes-fraction offsets from the corner; default -0.1/0.02)
      - pass explicit figure-fraction `x`/`y` for a label spanning multiple
        axes (a composite panel) — y is the panel's top edge.
    `va` defaults to 'top' (label sits just below y); pass va='bottom' to
    pair a label with a title that itself uses va='bottom' at the same y.
    """
    kwargs = dict(fontsize=FONT_SIZES['panel_label'], fontweight='bold', ha='left', va=va)
    if ax is not None:
        dx = -0.1 if dx is None else dx
        dy = 0.02 if dy is None else dy
        ax.text(dx, 1 + dy, label, transform=ax.transAxes, **kwargs)
    else:
        fig.text(x, y, label, **kwargs)


def save_figure(fig, out_path, formats=('pdf', 'png'), dpi=DPI):
    """Save figure to all requested formats, then close it.

    Each format is written into a subfolder named after the extension
    (e.g. pdfs/, pngs/, svgs/) inside the base output directory.

    Args:
        fig:      matplotlib Figure to save.
        out_path: Path with any extension — the extension is replaced per format.
        formats:  Tuple of format strings, e.g. ('pdf', 'png') or ('pdf', 'png', 'svg').
        dpi:      Resolution for raster formats.
    """
    base_dir = os.path.dirname(os.path.splitext(out_path)[0]) or '.'
    name     = os.path.splitext(os.path.basename(out_path))[0]
    saved = []
    for ext in formats:
        out_dir = os.path.join(base_dir, f"{ext}s")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches='tight', dpi=dpi, format=ext)
        saved.append(path)
    plt.close(fig)
    print(f"Saved: {', '.join(saved)}")
