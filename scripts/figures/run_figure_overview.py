import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cairosvg
import io
import os
import re
import shutil
import subprocess
import matplotlib
# from pdf2image import convert_from_path
import fitz  # pymupdf

from src.visualization.figure_base import apply_style, FONT_SIZES, FIGURE_WIDTH, save_figure

apply_style()



# --- Make matplotlib's DejaVu Sans available to cairo/fontconfig ---
# ~/.fonts is a standard user-level font dir, no sudo needed
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
print(f"Using SVG renderer: {_SVG_RENDERER}")


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

DrosoDoc = fitz.open('src/src_imgs/DrosoImaging.pdf')[0]
pix = DrosoDoc.get_pixmap(dpi=300)
DrosoImage = Image.open(io.BytesIO(pix.tobytes("png")))
# DrosoImage = Image.open('src/src_imgs/DrosoImaging.pdf')
RawData = Image.open('src/src_imgs/RawImages.png')
ExpHierarchy = load_svg('src/src_imgs/expHierarchy.svg')
ModelImage = load_svg('src/src_imgs/ModelArch.svg')
LatentSketch = load_svg('src/src_imgs/LatentSketch.svg')
placeholder = Image.open('src/src_imgs/Placeholder.png')


# --- Custom Layout ---
layout = [
    ['1', '2', '2', '3'],
    ['4', '4', '4', '5']
]

fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(FIGURE_WIDTH, 12),
    gridspec_kw={'width_ratios': [1, 1, 2, 2], 'height_ratios': [3, 5]}
)

labels = ['a', 'b', 'c', 'd', 'e']

for ax, label in zip(axes.values(), labels):
    ax.set_title(label, loc='left', fontsize=FONT_SIZES['panel_label'], color='black', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    if label == 'a':
        img = DrosoImage
    elif label == 'b':
        img = RawData
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

    for spine in ax.spines.values():
        spine.set_visible(False)


plt.tight_layout(pad=1)

# Shrink subplot 'e' by ~10%
pos = axes['5'].get_position()
shrink = 0.9
cx, cy = pos.x0 + pos.width / 2, pos.y0 + pos.height / 2
new_w, new_h = pos.width * shrink, pos.height * shrink
axes['5'].set_position([cx - new_w / 2, cy - new_h / 2, new_w, new_h])
save_figure(fig, 'results/CombiPlots/fig_overview.pdf', formats=('png', 'svg', 'pdf'), dpi=500)