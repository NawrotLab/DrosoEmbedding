import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np

# Create a figure with custom size
fig = plt.figure(figsize=(15, 10), dpi=300)

# Create a GridSpec with 2 rows and 2 columns
# Row 0 is 1/3 of the height, Row 1 is 2/3
# Col 0 is 2/3 of the width, Col 1 is 1/3
outer_gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[2, 1], hspace=0.1, wspace=0.1)

# Create inner GridSpec for row 0, col 0 (2 columns)
inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[0, 0], width_ratios=[1, 5])

# Add subplots
ax1 = fig.add_subplot(inner_gs[0, 0])
ax2 = fig.add_subplot(inner_gs[0, 1])
ax3 = fig.add_subplot(outer_gs[0, 1])
ax4 = fig.add_subplot(outer_gs[1, 0])
ax5 = fig.add_subplot(outer_gs[1, 1])

# Load placeholder image
img = Image.open('src/src_imgs/Placeholder.png')
DrosoImage = Image.open('src/src_imgs/DrosoImaging.png')
RawData = Image.open('src/src_imgs/RawImages_B_v0.png')
ExpHierarchy = Image.open('src/src_imgs/expHierarchy_C.png')
ModelImage = Image.open('src/src_imgs/ModelArch.png')
LatentSketch = Image.open('src/src_imgs/LatentSketch_v0.png')

# Display placeholder image in all subplots with labels
labels = ['A', 'B', 'C', 'D', 'E']


for ax, label in zip([ax1, ax2, ax3, ax4, ax5], labels):
    ax.axis('off')  # Turn off axes
    ax.text(0.05, 0.95, label, transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='top')
    if ax == ax1:
        ax.imshow(DrosoImage)
    elif ax == ax2:
        ax.imshow(RawData)
    elif ax == ax3:
        ax.imshow(ExpHierarchy)
    elif ax == ax4:
        ax.imshow(ModelImage)
    elif ax == ax5:
        ax.imshow(LatentSketch)
    else:
        ax.imshow(img)



plt.tight_layout(pad=0.5)
plt.savefig('results/CombiPlots/Figure1_v3_C.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
print("Figure saved to results/CombiPlots/Figure1_v3_C.png")