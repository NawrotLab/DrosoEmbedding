import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec
from PIL import Image
import cairosvg
import io
import os
import pickle
import re
import shutil
import subprocess
import matplotlib
import fitz  # pymupdf

from src.utils.config_loader import load_config
from src.visualization.figure_base import apply_style, FIGURE_WIDTH, save_figure, add_panel_label
from src.visualization.vizDataset_B import (
    load_and_process_data, get_time_points, draw_meanZ_row,
)
from src.analysis.example_frames import save_fig1_panel_b, load_fig1_panel_b, has_fig1_panel_b

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

def compute_panel_b_results():
    """Panel b's data-derived content: the 6 raw-LFM meanZ timepoints (per
    condition: Odor, Taste, Combi_M) that draw_meanZ_row() overlays on the
    static background. Needs paths['raw_recordings'] (one .nii per selected
    recording) and paths['peakIDs_Times_All']; same selection (random.seed(777)
    inside load_and_process_data) vizDataset_B.py's own standalone preview uses,
    so there is no separate manually-edited image asset to keep in sync with
    the actual pipeline.

    Returns (cond_names, rec_names, frames) -- frames: (n_cond, 6, H, W) float32,
    exactly what save_fig1_panel_b()/load_fig1_panel_b() store and load.
    """
    paths = load_config()['paths']
    lfm_path = paths['raw_recordings']
    with open(paths['peakIDs_Times_All'], 'rb') as f:
        pkl = pickle.load(f)
    panel_b_data = load_and_process_data(pkl, lfm_path)
    panel_b_time_points = get_time_points(pkl, panel_b_data)

    cond_names = list(panel_b_data.keys())
    rec_names = [panel_b_data[c]['recording'] for c in cond_names]
    # One (6, H, W) array per condition, NOT stacked into one tensor: each
    # condition is a different raw recording, and real recordings' raw pixel
    # dimensions vary (confirmed against Fig S1's own frames, which span
    # several distinct (H, W) shapes across recordings) -- stacking would
    # silently assume they match and crash (or worse, mis-align) when they don't.
    frames = [
        np.stack([panel_b_data[c]['meanZ'][:, :, t] for t in panel_b_time_points[c]], axis=0)
        .astype(np.float32)
        for c in cond_names
    ]
    return cond_names, rec_names, frames


def main():
    print(f"Using SVG renderer: {_SVG_RENDERER}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--recompute', action='store_true',
                        help='Ignore the stored panel-b results and recompute them '
                             '(needs paths[\'raw_recordings\'])')
    args = parser.parse_args()

    config = load_config()
    paths = config['paths']
    src_imgs_dir = paths['src_imgs_dir']
    out_path = os.path.join(paths['output_dir'], 'fig_overview.pdf')

    _register_dejavu_with_fontconfig()

    DrosoDoc = fitz.open(os.path.join(src_imgs_dir, 'DrosoImaging.pdf'))[0]
    pix = DrosoDoc.get_pixmap(dpi=300)
    DrosoImage = Image.open(io.BytesIO(pix.tobytes("png")))

    # Panel b results (a few small frames) live in the published results folder.
    # With the file present, no raw recordings are needed at all.
    panel_b_results_path = os.path.join(paths['results_dir'], 'example_frames.npz')
    if has_fig1_panel_b(panel_b_results_path) and not args.recompute:
        print(f'Loading stored panel-b results ({panel_b_results_path})')
        cond_names, rec_names, panel_b_frames = load_fig1_panel_b(panel_b_results_path)
    else:
        cond_names, rec_names, panel_b_frames = compute_panel_b_results()
        save_fig1_panel_b(panel_b_results_path, cond_names, rec_names, panel_b_frames)
        print(f'Saved panel-b results -> {panel_b_results_path}')
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
            # Explicit margins rather than tight_layout()'s automatic
            # guessing -- deterministic, won't silently drift if anything
            # else about the panels' content changes later.
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
    # placeholder axes with a nested GridSpec spanning its exact spot --
    # same top/bottom extent as panels a/c, no reserved margin. Separation
    # between rows comes entirely from hspace.
    b_subplotspec = axes['2'].get_subplotspec()
    axes['2'].remove()
    gs_b = GridSpecFromSubplotSpec(3, 1, subplot_spec=b_subplotspec, hspace=0.3)
    for i, cond in enumerate(cond_names):
        ax_row = fig.add_subplot(gs_b[i, 0])
        # (H, W, 6): matches draw_meanZ_row()'s meanZ_data[:, :, t] indexing convention.
        meanZ_stack = np.moveaxis(panel_b_frames[i], 0, -1)
        draw_meanZ_row(ax_row, meanZ_stack, list(range(panel_b_frames[i].shape[0])), panel_b_bg)

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
    add_panel_label(fig, 'b', x=lefts[1] - 0.01, y=tops[0])
    add_panel_label(fig, 'c', x=lefts[3], y=tops[0])
    add_panel_label(fig, 'd', x=lefts[0], y=row2_label_y)
    add_panel_label(fig, 'e', x=lefts[3], y=row2_label_y)

    save_figure(fig, out_path, formats=('png', 'svg', 'pdf'), dpi=500)


if __name__ == '__main__':
    main()