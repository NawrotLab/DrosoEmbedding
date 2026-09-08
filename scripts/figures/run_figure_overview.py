import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
from PIL import Image
import cairosvg
import io
import os
import pickle
import re
import shutil
import subprocess
import matplotlib
# from pdf2image import convert_from_path
import fitz  # pymupdf

from src.utils.config_loader import load_config
from src.visualization.figure_base import apply_style, FONT_SIZES, FIGURE_WIDTH, save_figure, add_panel_label
from src.visualization.vizDataset_B import (
    load_and_process_data, get_time_points, draw_meanZ_row,
)

apply_style()


# --- Make matplotlib's DejaVu Sans available to cairo/fontconfig ---
# ~/.fonts is a standard user-level font dir, no sudo needed
def _register_dejavu_with_fontconfig():
    user_font_dir = os.path.expanduser('~/.fonts')
    os.makedirs(user_font_dir, exist_ok=True)
    mpl_font_dir = os.path.dirname(matplotlib.font_manager.findfont('DejaVu Sans'))
    for f in os.listdir(mpl_font_dir):
        if f.lower().endswith(('.ttf', '.otf')):
            src = os.path.join(mpl_font_dir, f)
            dst = os.path.join(user_font_dir, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    # Rebuild fontconfig cache for the user
    subprocess.run(['fc-cache', '-f', user_font_dir], check=False)


def _find_svg_renderer():
    """Find the best available SVG renderer."""
    for cmd in ['inkscape', 'rsvg-convert']:
        if shutil.which(cmd):
            return cmd
    return 'cairosvg'

_SVG_RENDERER = _find_svg_renderer()


def load_svg(path, dpi=1000):
    """Rasterise an SVG file and return it as a PIL Image."""
    path = os.path.abspath(path)

    # Replace math-italic theta variants (from PowerPoint/Word equation editor)
    # with standard Greek θ (U+03B8) which most fonts support
    with open(path, 'rb') as f:
        svg_data = f.read()
    math_thetas = [
        b'\xf0\x9d\x9c\x83',  # U+1D703 MATHEMATICAL ITALIC SMALL THETA
        b'\xf0\x9d\x9b\xb3',  # U+1D6F3 MATHEMATICAL BOLD SMALL THETA
        b'\xf0\x9d\x9c\xbd',  # U+1D73D MATHEMATICAL BOLD ITALIC SMALL THETA
        b'\xf0\x9d\x9d\x97',  # U+1D777 MATHEMATICAL SANS-SERIF BOLD SMALL THETA
        b'\xf0\x9d\x9e\x51',  # U+1D7B1 MATHEMATICAL SANS-SERIF BOLD ITALIC SMALL THETA
        b'\xce\xb8',          # U+03B8 standard theta (keep as-is, but listed for reference)
    ]
    standard_theta = b'\xce\xb8'  # UTF-8 for θ (U+03B8)
    for mt in math_thetas[:-1]:  # skip the last one (already standard)
        svg_data = svg_data.replace(mt, standard_theta)

    if _SVG_RENDERER == 'inkscape':
        # Write modified SVG to temp file for inkscape
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix='.svg', delete=False)
        tmp.write(svg_data); tmp.close()
        result = subprocess.run(
            ['inkscape', tmp.name, '--export-type=png', '--export-filename=-',
             f'--export-dpi={dpi}'],
            capture_output=True
        )
        os.unlink(tmp.name)
        return Image.open(io.BytesIO(result.stdout))

    elif _SVG_RENDERER == 'rsvg-convert':
        result = subprocess.run(
            ['rsvg-convert', '-d', str(dpi), '-p', str(dpi)],
            input=svg_data, capture_output=True
        )
        return Image.open(io.BytesIO(result.stdout))

    else:  # cairosvg fallback
        # Replace font-family in XML attributes
        svg_data = re.sub(rb'font-family="[^"]*"', b'font-family="DejaVu Sans"', svg_data)
        svg_data = re.sub(rb"font-family='[^']*'", b"font-family='DejaVu Sans'", svg_data)
        # Replace font-family inside style="..." attributes (inline CSS)
        svg_data = re.sub(
            rb'(style="[^"]*?)font-family\s*:\s*[^;"]+',
            rb'\1font-family: DejaVu Sans', svg_data)
        svg_data = re.sub(
            rb"(style='[^']*?)font-family\s*:\s*[^;']+",
            rb"\1font-family: DejaVu Sans", svg_data)
        style = b'<style>* { font-family: "DejaVu Sans", sans-serif !important; }</style>'
        svg_data = re.sub(rb'(<svg[^>]*>)', rb'\1' + style, svg_data, count=1)
        png_data = cairosvg.svg2png(bytestring=svg_data, dpi=dpi)
        return Image.open(io.BytesIO(png_data))

def main():
    print(f"Using SVG renderer: {_SVG_RENDERER}")

    config = load_config()
    paths = config['paths']
    src_imgs_dir = paths['src_imgs_dir']
    out_path = os.path.join(paths['output_dir'], 'fig_overview.pdf')

    _register_dejavu_with_fontconfig()

    DrosoDoc = fitz.open(os.path.join(src_imgs_dir, 'DrosoImaging.pdf'))[0]
    pix = DrosoDoc.get_pixmap(dpi=300)
    DrosoImage = Image.open(io.BytesIO(pix.tobytes("png")))

    # Panel b: built live from raw LFM data (Odor/Taste/Combi_M), same as
    # vizDataset_B.py's standalone preview -- see draw_meanZ_row() there,
    # shared rather than duplicated, so there's no separate manually-edited
    # image asset to keep in sync with the actual pipeline.
    lfm_path = os.path.join(paths['data_root'], 'Paul_LFM_Data')
    with open(paths['peakIDs_Times_All'], 'rb') as f:
        pkl = pickle.load(f)
    panel_b_data = load_and_process_data(pkl, lfm_path)
    panel_b_time_points = get_time_points(pkl, panel_b_data)
    panel_b_bg = mpimg.imread(os.path.join(src_imgs_dir, 'RawImages_blank_B.png'))

    ExpHierarchy = load_svg(os.path.join(src_imgs_dir, 'expHierarchy.svg'))
    ModelImage = load_svg(os.path.join(src_imgs_dir, 'ModelArch.svg'))
    LatentSketch = load_svg(os.path.join(src_imgs_dir, 'LatentSketch.svg'))
    placeholder = Image.open(os.path.join(src_imgs_dir, 'Placeholder.png'))

    # --- Custom Layout ---
    layout = [
        ['1', '2', '2', '3'],
        ['4', '4', '4', '5']
    ]

    fig, axes = plt.subplot_mosaic(
        layout,
        figsize=(FIGURE_WIDTH, 12),
        gridspec_kw={
            'width_ratios': [1, 1, 2, 2], 'height_ratios': [3, 5],
            # Explicit margins instead of tight_layout()'s automatic guessing
            # (removed below) -- deterministic, won't silently drift if
            # anything else about the panels' content changes later.
            'left': 0.04, 'right': 0.98,
            'top': 0.90, 'bottom': 0.04,
            'wspace': 0.12, 'hspace': 0.15,
        }
    )

    labels = ['a', 'b', 'c', 'd', 'e']

    for ax, label in zip(axes.values(), labels):
        if label == 'b':
            continue  # built separately below as a nested 3-row sub-grid

        ax.set_xticks([]); ax.set_yticks([])

        if label == 'a':
            img = DrosoImage
        elif label == 'c':
            img = ExpHierarchy
        elif label == 'd':
            img = ModelImage
        elif label == 'e':
            img = LatentSketch
        else:
            img = placeholder

        ax.imshow(img, aspect='equal')
        ax.set_xlim(0, img.size[0])
        ax.set_ylim(img.size[1], 0)
        if label in ('d', 'e'):
            # d/e's images are much wider than tall relative to their box,
            # so centering (default) leaves a big empty gap above the image,
            # between it and the label. Anchor to the top instead -- safe to
            # do now since labels are positioned from the nominal grid, not
            # from each axes' own post-aspect box (see below), so this can't
            # un-align them the way it did before.
            ax.set_anchor('N')

        for spine in ax.spines.values():
            spine.set_visible(False)

    # Shrink subplot 'e' by ~10%
    pos = axes['5'].get_position()
    shrink = 0.9
    cx, cy = pos.x0 + pos.width / 2, pos.y0 + pos.height / 2
    new_w, new_h = pos.width * shrink, pos.height * shrink
    axes['5'].set_position([cx - new_w / 2, cy - new_h / 2, new_w, new_h])

    # Panel b: 3 stacked rows (Odor, Taste, Combi_M), replacing the single
    # placeholder axes with a nested GridSpec in its exact spot.
    b_subplotspec = axes['2'].get_subplotspec()
    axes['2'].remove()
    # 4 rows, not 3: row 0 is a small empty spacer so the first data row's
    # top-anchored image (aspect='auto' fills its whole box) doesn't touch
    # label 'b' (GridSpecFromSubplotSpec has no top/bottom margin params --
    # only the top-level GridSpec does). No header text anymore, so this
    # only needs to be tall enough for that clearance, not a title.
    gs_b = GridSpecFromSubplotSpec(
        4, 1, subplot_spec=b_subplotspec, hspace=0.15,
        height_ratios=[0.15, 1, 1, 1],
    )
    b_row_axes = []
    for i, cond in enumerate(panel_b_data.keys()):
        ax_row = fig.add_subplot(gs_b[i + 1, 0])
        draw_meanZ_row(ax_row, panel_b_data[cond]['meanZ'], panel_b_time_points[cond], panel_b_bg)
        b_row_axes.append(ax_row)

    # Panel labels, placed after layout is finalised, all from the shared
    # GridSpec's nominal row/column boundaries rather than each axes' own
    # (post aspect='equal') box: imshow(aspect='equal') on differently-shaped
    # images shrinks each axes' box individually around its own center, so
    # per-axes anchoring (ax.transAxes) made labels that should align (a/b/c
    # share a row; a/d and c/e share a column) drift apart even though the
    # underlying grid geometry is identical. d/e share one y a bit below the
    # nominal row-1 top -- 'e' is additionally shrunk 10% around its own
    # center (above), which pulls its own box's top down from that nominal
    # line, so anchoring both to the same explicit y aligns them exactly.
    gs = axes['1'].get_gridspec()
    _, tops, lefts, _ = gs.get_grid_positions(fig)
    row2_label_y = tops[1] + 0.015
    add_panel_label(fig, 'a', x=lefts[0], y=tops[0])
    add_panel_label(fig, 'b', x=lefts[1] + 0.01, y=tops[0])
    add_panel_label(fig, 'c', x=lefts[3], y=tops[0])
    add_panel_label(fig, 'd', x=lefts[0], y=row2_label_y)
    add_panel_label(fig, 'e', x=lefts[3], y=row2_label_y)

    save_figure(fig, out_path, formats=('png', 'svg', 'pdf'), dpi=500)


if __name__ == '__main__':
    main()