import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

placeholder = Image.open('src/src_imgs/Placeholder.png')
DrosoImage = Image.open('src/src_imgs/DrosoImaging.png')
RawData = Image.open('src/src_imgs/RawImages_B_v0.png')
ExpHierarchy = Image.open('src/src_imgs/expHierarchy_C.png')
ModelImage = Image.open('src/src_imgs/ModelArch.png')
LatentSketch = Image.open('src/src_imgs/LatentSketch_v0.png')


# --- Custom Layout ---
layout = [
    ['1', '2', '2', '3'],
    ['4', '4', '4', '3'],
    ['4', '4', '4', '5']
]

fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(12, 8),
    gridspec_kw={'width_ratios': [1, 2, 2, 2], 'height_ratios': [3, 1, 2]}
)
# fig.patch.set_facecolor('lightgrey')  # Figure background

labels = ['a', 'b', 'c', 'd', 'e']
# labels = ['A', 'B', 'C', 'D', 'E']

for ax, label in zip(axes.values(), labels):
    ax.set_title(label, loc='left', fontsize=14, color='black')
    ax.set_xticks([]); ax.set_yticks([])
    if label == 'a':
        ax.imshow(DrosoImage, aspect='auto')
    elif label == 'b':
        ax.imshow(RawData, aspect='auto')
    elif label == 'c':
        ax.imshow(ExpHierarchy, aspect='auto')
    elif label == 'd':
        ax.imshow(ModelImage, aspect='auto')
    elif label == 'e':
        ax.imshow(LatentSketch, aspect='auto')
    else:
        ax.imshow(placeholder, aspect='auto')
    # REMOVE borders:
    for spine in ax.spines.values():
        spine.set_visible(False)


plt.tight_layout(pad=1)
plt.savefig('results/CombiPlots/Figure1_v4.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
print("Figure saved to results/CombiPlots/Figure1_v4.png")

