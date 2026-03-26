import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DrosoImage = Image.open('src/src_imgs/DrosoImaging.png')
RawData = Image.open('src/src_imgs/RawImages.png')
ExpHierarchy = Image.open('src/src_imgs/expHierarchy.png')
ModelImage = Image.open('src/src_imgs/ModelArch.png')
LatentSketch = Image.open('src/src_imgs/LatentSketch.png')
placeholder = Image.open('src/src_imgs/Placeholder.png')


# --- Custom Layout ---
# layout = [
#     ['1', '2', '2', '3'],
#     ['4', '4', '4', '3'],
#     ['4', '4', '4', '5']
# ]
# fig, axes = plt.subplot_mosaic(
#     layout,
#     figsize=(12, 8),
#     gridspec_kw={'width_ratios': [1, 2, 2, 2], 'height_ratios': [3, 1, 2]} 
# )

layout = [
    ['1', '2', '2', '3'],
    ['4', '4', '4', '5']
]

fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(12, 8),
    # gridspec_kw={'width_ratios': [1, 2, 2, 2], 'height_ratios': [3, 1, 2]} 
    gridspec_kw={'width_ratios': [1, 1, 2, 2], 'height_ratios': [3,5]}
)
# fig.patch.set_facecolor('lightgrey')  # Figure background

labels = ['a', 'b', 'c', 'd', 'e']
# labels = ['A', 'B', 'C', 'D', 'E']

# for ax, label in zip(axes.values(), labels):
#     # print(ax.get_position())
#     # x1_pos = ax.get_position().x1 
#     # y1_pos = ax.get_position().y1
#     # fig.text(x1_pos, y1_pos, label, fontsize=10, ha='left', va='bottom', color='black')

#     ax.set_title(label, loc='left', fontsize=10, color='black', fontweight='bold')
#     ax.set_xticks([]); ax.set_yticks([])
#     if label == 'a':
#         ax.imshow(DrosoImage, aspect='auto')
#     elif label == 'b':
#         ax.imshow(RawData, aspect='auto')
#     elif label == 'c':
#         ax.imshow(ExpHierarchy, aspect='auto')
#     elif label == 'd':
#         ax.imshow(ModelImage, aspect='auto')
#     elif label == 'e':
#         ax.imshow(LatentSketch, aspect = 'equal') # , aspect='auto'
#     else:
#         ax.imshow(placeholder, aspect='auto')
    
#     # remove borders:
#     for spine in ax.spines.values():
#         spine.set_visible(False)

for ax, label in zip(axes.values(), labels):
    ax.set_title(label, loc='left', fontsize=10, color='black', fontweight='bold')
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

    # remove borders:
    for spine in ax.spines.values():
        spine.set_visible(False)



plt.tight_layout(pad=1)
plt.savefig('results/CombiPlots/fig_overview_v1.png', dpi=500, bbox_inches='tight', pad_inches=0.1)

