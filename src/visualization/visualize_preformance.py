import matplotlib.pyplot as plt
import torch
import numpy as np 
from sklearn.manifold import TSNE
import seaborn as sns
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
import textwrap

import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# evaluation/cam_utils.py
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np

# If you're using jacobgil/pytorch-grad-cam (recommended):
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# visualization/cam_plots.py
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
Tensor = torch.Tensor

# def get_class_style(class_name, styles):
#     """Get style properties for a given class name.
    
#     Args:
#         class_name: The name of the class to get style for
#         styles: Dictionary of style properties from stylesE.yaml
        
#     Returns:
#         Dictionary with style properties or None if not found
#     """
#     if not class_name or not styles:
#         return None
        
#     # Try exact match first
#     if class_name in styles:
#         return styles[class_name]
    
#     # Normalize the class name for matching
#     normalized = class_name.lower().strip()
    
#     # Try different variations of the class name
#     variations = [
#         normalized,
#         normalized.replace(' ', '_'),
#         normalized.replace('-', '_'),
#         normalized.replace(' ', ''),
#         normalized.replace('-', ''),
#         normalized.replace('_', ''),
#         normalized.replace(' ', '_').replace('-', '_'),
#     ]
    
#     # Try to find a matching style
#     for variation in variations:
#         if variation in styles:
#             return styles[variation]
    
#     # Try partial matches (e.g., 'fed_odor' matches 'fed_odor_positive')
#     for key, style in styles.items():
#         if isinstance(key, str) and key in normalized:
#             return style
    
#     # Try common variations for positive/negative
#     if 'pos' in normalized or '+' in normalized:
#         return styles.get('positive') or styles.get('pos')
#     if 'neg' in normalized or '-' in normalized:
#         return styles.get('negative') or styles.get('neg')
    
#     # No match found
#     return None


def _plot_class_symbol(ax, x, y, style, markersize=12, transform=None, clip_on=False, zorder=10):
    """Draw a class symbol (regular dot or bicolor split-circle) at (x, y)."""
    from src.utils.helpers import HALF_CIRCLE_LEFT, HALF_CIRCLE_RIGHT
    kw = dict(clip_on=clip_on, zorder=zorder)
    if transform is not None:
        kw['transform'] = transform

    if style.get('bicolor', False):
        ax.plot(x, y, marker=HALF_CIRCLE_LEFT, markersize=markersize,
                markerfacecolor=style.get('left_color', 'gray'),
                markeredgecolor=style.get('left_edgecolor', 'black'), **kw)
        ax.plot(x, y, marker=HALF_CIRCLE_RIGHT, markersize=markersize,
                markerfacecolor=style.get('right_color', 'gray'),
                markeredgecolor=style.get('right_edgecolor', 'black'), **kw)
    else:
        ax.plot(x, y, marker=style.get('shape', 'o'), markersize=markersize,
                markerfacecolor=style.get('color', 'gray'),
                markeredgecolor=style.get('edgecolor', 'black'), **kw)



def get_class_style(class_name, styles):
    """Get style properties for a given class name.
    
    Args:
        class_name: The name of the class to get style for
        styles: Dictionary of style properties from stylesE.yaml
        
    Returns:
        Dictionary with style properties or None if not found
    """
    if not class_name or not styles:
        return None
        
    # First, try to find a direct match in the styles
    for style_key, style in styles.items():
        # Check if this style has a class_name that matches
        if 'class_name' in style and style['class_name'] == class_name:
            print(f"Found direct match for {class_name}: {style}")
            return style
    
    # If no direct match found, try matching with variations of the class name
    normalized = class_name.lower().strip()
    
    # Generate possible variations of the class name for matching
    variations = [
        normalized,
        normalized.replace(' ', '_'),
        normalized.replace('-', '_'),
        normalized.replace(' ', ''),
        normalized.replace('-', ''),
        normalized.replace('_', ''),
        normalized.replace(' ', '_').replace('-', '_'),
    ]
    
    # Try to find a matching style
    for variation in variations:
        if variation in styles:
            print(f"Found variation match for {class_name}: {styles[variation]}")
            return styles[variation]
    
    return None

def plot_weighted_avg(report_dict, output_path):
    metrics = ["accuracy", "precision", "recall", "f1-score"]
    values = [
        report_dict["accuracy"], 
        report_dict["weighted avg"]["precision"], 
        report_dict["weighted avg"]["recall"], 
        report_dict["weighted avg"]["f1-score"]
    ]

    plt.figure(figsize=(5, 4))
    plt.bar(metrics, values, color=["#8c564b", "#4e79a7", "#f28e2b", "#59a14f"])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Weighted Average + Accuracy")
    plt.tight_layout()

    out_path = os.path.join(output_path, "weighted_avg.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_per_class_metrics(report_dict, output_path):
    classes = [k for k in report_dict.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
    metrics = ["precision", "recall", "f1-score"]
    num_classes = len(classes)
    chance_level = 1.0 / num_classes

    values = {m: [report_dict[cls][m] for cls in classes] for m in metrics}
    x = np.arange(num_classes)
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, values["precision"], width, label="Precision")
    plt.bar(x,         values["recall"],    width, label="Recall")
    plt.bar(x + width, values["f1-score"],  width, label="F1-score")

    # Add accuracy line with annotation
    accuracy = report_dict["accuracy"]
    plt.axhline(accuracy, color="gray", linestyle="--", label="Accuracy")
    plt.text(num_classes - 0.5, accuracy + 0.01, f"Accuracy: {accuracy:.2f}", 
             color="gray", fontsize=9, va="bottom", ha="right")

    # Adjust y-axis
    y_min = np.floor(chance_level * 10) / 10
    plt.ylim(y_min, 1.0)
    plt.yticks(np.arange(y_min, 1.01, 0.1))

    plt.xticks(x, classes, rotation=45)
    plt.ylabel("Score")
    plt.title("Per-Class Performance")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_path, "per_class_metrics.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_train_val_loss(training_loss, validation_loss, dataID, model_name,
                        batch_size, learning_rate, output_path, ylim=None,
                        training_acc=None, validation_acc=None):
    """
    Plot and save the training and validation loss and accuracy curves.

    Args:
    - training_loss: List of training loss values per epoch.
    - validation_loss: List of validation loss values per epoch.
    - dataID: Identifier for the dataset.
    - model_name: Name of the model used.
    - batch_size: Batch size used during training.
    - learning_rate: Learning rate used during training.
    - output_path: Directory where the plot will be saved.
    - ylim: Optional tuple for y-axis limits for loss.
    - training_acc: Optional list of training accuracy values per epoch.
    - validation_acc: Optional list of validation accuracy values per epoch.
    """
    
    num_epochs = len(training_loss)
    title = f"{model_name}: {dataID}"
    subtitle = f"BS: {batch_size}; LR: {learning_rate}; Epochs: {num_epochs}"

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot loss on primary y-axis (left)
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('Cross Entropy Loss', color='darkslategray', fontsize=16)
    loss_train, = ax1.plot(range(1, num_epochs + 1), training_loss, 
                          color='darkslategray', label='Training Loss', linewidth=2)
    loss_val, = ax1.plot(range(1, num_epochs + 1), validation_loss, 
                        color='cadetblue', label='Validation Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='darkslategray', labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    
    if ylim:
        ax1.set_ylim(ylim)
    
    # Create second y-axis for accuracy if accuracy data is provided
    if training_acc is not None and validation_acc is not None:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color='darkred', fontsize=16)
        acc_train, = ax2.plot(range(1, num_epochs + 1), training_acc, 
                             color='lightcoral', linestyle='--', 
                             label='Training Accuracy', linewidth=2)
        acc_val, = ax2.plot(range(1, num_epochs + 1), validation_acc, 
                           color='indianred', linestyle='--', 
                           label='Validation Accuracy', linewidth=2)
        ax2.tick_params(axis='y', labelcolor='darkred', labelsize=14)
        
        # Combine legends from both axes
        lines = [loss_train, loss_val, acc_train, acc_val]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, fontsize=12, loc='upper center', 
                  bbox_to_anchor=(0.5, -0.15), ncol=2)
    else:
        ax1.legend(fontsize=12, loc='upper center', 
                  bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    plt.tight_layout()
    classes = dataID.split("_")[2:]
    out_path = os.path.join(output_path, f"{len(classes)}Cls_LossAccuracyPlot.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Train/Validation loss and accuracy plot saved to {out_path}")
    return out_path

def plot_loss_accuracy(training_loss, validation_loss, training_accuracy,  dataID, model_name,
                        batch_size, learning_rate, output_path, ylim=None):
    num_epochs = len(training_loss)


    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), training_loss, c='darkslategray', label='Training Loss', linewidth=2)
    plt.plot(range(1, num_epochs + 1), validation_loss, c = 'cadetblue', label='Validation Loss', linewidth=2)
    
    # plt.xlim([0,250])
    plt.xlabel('Epoch', fontsize=16)
    plt.xticks(fontsize=14)  # Adjusts x-axis tick size
    plt.yticks(fontsize=14)  # Adjusts y-axis tick size
    plt.ylabel('Cross Entropy Loss', fontsize=16)
    # plt.axvline(x=250, linestyle="--", color="black", linewidth=2)  # Dashed vertical line at x=5
    plt.legend(fontsize=14)
    if ylim:
        plt.ylim(ylim)
    # plt.title(subtitle, fontsize=14)
    # plt.suptitle(title, fontsize=16)
    classes = dataID.split("_")[2:]
    out_path = os.path.join(output_path, f"{len(classes)}Cls_LossPlot.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Train/Validation loss plot saved to {out_path}")
            


def plot_confusion_matrix(cl_name, cm, class_names, output_path, dataID, hyperparameters=None, ax=None, annot=True, cbar=False, 
                        axis_labeling='both', use_class_symbols=False, styles=None):
    """
    Plot and save a confusion matrix with overall accuracy in the title.

    Args:
    - cl_name: Classifier name or title.
    - cm: Confusion matrix array.
    - class_names: List of class names for the matrix.
    - output_path: Directory where the plot will be saved. If None, returns the Axes.
    - dataID: Identifier for the dataset.
    - hyperparameters: Optional dict of hyperparameters for the title.
    - ax: Optional matplotlib Axes to draw on. If None, creates a new figure/axes.
    - annot: Boolean flag to enable cell annotations (percentages only).
    - cbar: Boolean flag to draw a colorbar for this matrix.
    - axis_labeling: Controls which axis labels are shown ('both', 'x_axis', 'y_axis', 'none')
    - use_class_symbols: If True, use colored dots instead of text labels for class names
    - styles: Dictionary mapping class names to style properties (color, edgecolor, shape))
    """
    save_fig = output_path is not None
    standalone = ax is None

    # Create or use provided Axes
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 12))
        # font sizes for standalone figure
        if len(class_names) < 4:
            fontsize_annot = 30
            fontsize_ticks = 30
            fontsize_axis = 30
            fontsize_title = 30
        else:
            fontsize_annot = 12
            fontsize_ticks = 12
            fontsize_axis = 18
            fontsize_title = 26
    else:
        fig = ax.figure
        fontsize_annot = 12
        fontsize_ticks = 12
        fontsize_axis = None
        fontsize_title = None

    # Compute overall accuracy
    total_correct = np.trace(cm)
    total_samples = np.sum(cm)
    accuracy = (total_correct / total_samples) * 100

    # Title only in standalone mode
    if standalone:
        hp_str = ', '.join(f"{k}={v}" for k, v in (hyperparameters or {}).items())
        title = f"Test Set Accuracy: {accuracy:.2f}%"
        if hp_str:
            title += f" ({hp_str})"
        wrapped_title = "\n".join(textwrap.wrap(title, width=70))
        ax.set_title(wrapped_title, fontsize=fontsize_title, pad=20)

    # Normalize to percentages
    cm_pct = (cm / cm.sum(axis=1, keepdims=True)) * 100

    # Prepare annotations
    if annot:
        annot_data = np.array([
            [f"{pct:.1f}%" for pct, _ in zip(row_pct, row_cnt)]
            for row_pct, row_cnt in zip(cm_pct, cm)
        ])
        annot_kws = {"size": fontsize_annot} if fontsize_annot else {}
    else:
        annot_data = False
        annot_kws = {}

    # Handle axis labeling based on parameter
    if use_class_symbols and styles:
        # For styled dots, we'll handle the labels separately
        xticklabels = [''] * len(class_names)
        yticklabels = [''] * len(class_names)
    else:
        # Use text labels only
        if axis_labeling == 'both':
            xticklabels = class_names
            yticklabels = class_names if standalone else []
        elif axis_labeling == 'x_axis':
            xticklabels = class_names
            yticklabels = []
        elif axis_labeling == 'y_axis':
            xticklabels = []
            yticklabels = class_names if standalone else []
        else:  # 'none' or None
            xticklabels = []
            yticklabels = []

    # Draw heatmap
    sns.heatmap(
        cm_pct,
        annot=annot_data,
        fmt="",
        cmap='Blues',
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cbar=cbar,
        ax=ax,
        annot_kws=annot_kws,
        vmin=0,
        vmax=100
    )
    
    # Add styled dots for class labels if enabled
    if use_class_symbols and styles:
        # Get the style for each class
        for i, class_name in enumerate(class_names):
            style = get_class_style(class_name, styles)
            if style:
                # Add dot for y-axis (row) label
                if axis_labeling in ['both', 'y_axis']:
                    _plot_class_symbol(ax, -0.05, i + 0.5, style, markersize=12,
                                       transform=ax.get_yaxis_transform(),
                                       clip_on=False, zorder=10)

                # Add dot for x-axis (column) label
                if axis_labeling in ['both', 'x_axis']:
                    _plot_class_symbol(ax, i + 0.5, -0.04, style, markersize=12,
                                       transform=ax.get_xaxis_transform(),
                                       clip_on=False, zorder=10)

    n_classes = len(class_names)
    
    # Calculate tick positions for centering
    xticks = np.arange(n_classes) + 0.5  # Centered horizontally
    yticks = np.arange(n_classes) + 0.5  # Centered vertically
    
    # Set ticks
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', length=2)  # Remove tick marks but keep grid lines
    
    # Set grid lines
    # ax.grid(True, which='major', axis='both', linestyle='-', color='w', linewidth=1)
    
    # For 'none' case, hide labels but keep ticks
    if axis_labeling == 'none':
        ax.set_xticklabels([])
        ax.set_yticklabels([])


    # Labels in standalone mode
    if standalone:
        ax.set_xlabel('Predicted Label', fontsize=fontsize_axis, labelpad=20)
        ax.set_ylabel('True Label', fontsize=fontsize_axis, labelpad=20)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Rotate ticks
    ax.tick_params(axis='x', rotation=45, labelsize=fontsize_ticks or plt.rcParams['xtick.labelsize'])
    if standalone:
        ax.tick_params(axis='y', rotation=90, labelsize=fontsize_ticks)
    
    # If no labels, keep ticks but remove labels
    if axis_labeling == 'none':
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    if save_fig and standalone:
        os.makedirs(output_path, exist_ok=True)
        out_path = os.path.join(output_path, f"{len(class_names)}Cls_ConfusionMatrix.png")
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.3, dpi=300)
        plt.close(fig)
        print(f"Confusion matrix saved to {out_path}")
    else:
        return ax

def dataloader2dictionary(dataloader, classes_namelist):

    # Convert dataloader into dictionary with respective classes
    data_by_class = {class_name: [] for class_name in classes_namelist}
    for inputs, labels in dataloader:
        for input_tensor, label_idx in zip(inputs, labels):
            label = classes_namelist[label_idx]
            data_by_class[label].append(input_tensor)
    data_by_class = {class_id: torch.stack(tensors) for class_id, tensors in data_by_class.items()}
    return data_by_class


def plot_tsne_latent(latent_2d,
                     labels_np,
                     class_names,
                     output_path=None,
                     filename="latent_tsne.png",
                     xlim=None,
                     ylim=None,
                     colors=None,
                     shapes=None,
                     edgecolors=None,
                     bicolor_info=None,
                     ax=None,
                     title=None,
                     legend=True,
                     draw_axis=True,
                     draw_title=True):
    """
    Scatter‐plot of a precomputed TSNE embedding.

    Args:
      latent_2d: (N,2) array of t-SNE coordinates
      labels_np: (N,) array of class labels
      class_names: list of class name strings
      output_path: if provided, save standalone figure here
      filename: name for saving
      bicolor_info: dict mapping class_index -> {left_color, right_color, left_edgecolor, right_edgecolor}
      ax: Optional Axes to draw on (subplot mode)
      title: Optional subplot title
      legend: whether to show legend
      draw_axis: whether to draw x/y labels
      draw_title: whether to draw the title
    """
    from src.utils.helpers import scatter_bicolor_cloud, HALF_CIRCLE_LEFT

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 10))
    if colors is None:
        colors = sns.color_palette("tab20", len(class_names))
    if shapes is None:
        shapes = ['o'] * len(class_names)
    if edgecolors is None:
        edgecolors = ['none'] * len(class_names)
    if bicolor_info is None:
        bicolor_info = {}

    for cls in np.unique(labels_np):
        ci = int(cls)
        idx = labels_np == cls
        lbl = f"{class_names[ci]} ({idx.sum()})"

        if ci in bicolor_info:
            # Render bicolor split-circle cloud
            scatter_bicolor_cloud(ax, latent_2d[idx, 0], latent_2d[idx, 1],
                                  bicolor_info[ci], s=6, alpha=0.8, linewidth=0.3)
            # Invisible scatter just for legend entry
            ax.scatter([], [], marker=HALF_CIRCLE_LEFT,
                       c=[bicolor_info[ci]['left_color']],
                       edgecolors=[bicolor_info[ci]['left_edgecolor']],
                       s=20, label=lbl)
        else:
            ax.scatter(latent_2d[idx, 0], latent_2d[idx, 1],
                       label=lbl,
                       alpha=0.8, s=6,
                       color=colors[ci], edgecolors=edgecolors[ci],
                       linewidth=0.5, marker=shapes[ci])
    ax.plot(0, 0, 'ko', markersize=3)

    if xlim:
        xmin, xmax = xlim
    else: 
        xmin, xmax = ax.get_xlim()
    if ylim:
        ymin, ymax = ylim
    else:
        ymin, ymax = ax.get_ylim()
    ax.set_xlim(xmin, xmax) 
    ax.set_ylim(ymin, ymax)
    

    # determine limits
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', xmin))
    ax.spines['bottom'].set_position(('data', ymin))


    if draw_axis:
        ax.set_xlabel('TSNE Component 1', fontsize=14)
        ax.set_ylabel('TSNE Component 2', fontsize=14)
    if draw_title and title:
        ax.set_title(title, fontsize=16, pad=10)
    if legend:
        ax.legend(loc='upper right', fontsize=12)

    if standalone and output_path:
        os.makedirs(output_path, exist_ok=True)
        out = os.path.join(output_path, filename)
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"TSNE plot saved to {out}")
        return fig, ax

    return ax


def plot_umap_latent(latent_2d,
                     labels_np,
                     class_names,
                     output_path=None,
                     filename="latent_umap.png",
                     ax=None,
                     title=None):
    """
    Scatter‐plot of a precomputed UMAP embedding.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 10))

    for class_label in np.unique(labels_np):
        idx = labels_np == class_label
        count = idx.sum()
        ax.scatter(latent_2d[idx, 0],
                   latent_2d[idx, 1],
                   label=f"{class_names[int(class_label)]} ({count})",
                   alpha=0.5, s=50,
                   color=sns.color_palette("tab20")[int(class_label)],
                   edgecolors='gray', linewidth=0.5)

    ax.set_xlabel('UMAP Component 1', fontsize=26)
    ax.set_ylabel('UMAP Component 2', fontsize=26)
    if title:
        ax.set_title(title, fontsize=28)
    ax.legend(loc='upper right', fontsize=16, markerscale=2)

    if standalone and output_path:
        out = os.path.join(output_path, filename)
        fig.savefig(out)
        plt.close(fig)
        print(f"UMAP plot saved to {out}")
        return fig, ax

    return ax


# def reshape_transform4Cam(tensor):
#     # Handle tensors with 5 dimensions (e.g., from 3D models)
#     # print(f"Original tensor shape: {tensor.shape}")  # Debugging

#     if tensor.dim() == 5:  # Example shape: [B, C, D, H, W]
#         # Collapse depth dimension (D), returning to [B, C, H, W]
#         tensor = tensor.mean(dim=2)  # Take mean across the depth dimension
#     # print(f"Transformed tensor shape: {tensor.shape}")  # Debugging

#     # Ensure tensor is 4D: [N, C, H, W]
#     return tensor



# def plot_classes_cam(model, device, dataloader, class_names, output_path, cam_method = GradCAM, classifier_target_layer = 3, reshape = True):


#     # 1. Convert dataloader into dictionary with respective classes
#     data_by_class = dataloader2dictionary(dataloader, class_names)

#     # 2. Plot Input Tensors.
#     plot_InputTensors(data_by_class, output_path, 'Mean_InputTensors')
#     print(f'Mean Input Tensors saved: {output_path}')

#     # 3. Set Model up for Evaluation and initiate cam
#     model.eval()
#     target_layer =  model.cnn[classifier_target_layer]
#     if reshape:
#         cam = cam_method(model=model, target_layers=[target_layer], reshape_transform=reshape_transform4Cam)
#     else:
#         cam = cam_method(model=model, target_layers=[target_layer])

#     # 4. Get all heatmaps and sort in a dictionary
#     heatmaps = {class_name: [] for class_name in class_names}
#     for class_idx, (class_label, class_tensor) in enumerate(data_by_class.items()):
#         for image in class_tensor: #Shape: 5,1,128,128
#             if len(image.shape) == 3:
#                 image = image.unsqueeze(0).float().to(device)  # Shape: 1,1,128,128
#                 # image = image.unsqueeze(0).unsqueeze(0).float().to(device) #Shape: 1,1,1,128,128
#             else:
#                 image = image[0:1].float().to(device)  # Shape: 1,1,128,128

#                 # image = image[0, :, :, :].unsqueeze(0).unsqueeze(0).float().to(device) #Shape: 1,1,1,128,128
#             heatmap = cam(input_tensor=image, targets=[ClassifierOutputTarget(class_idx)]) # class_label
#             heatmap_2D = heatmap[0]
#             heatmaps[class_label].append(heatmap_2D) #result_img

#     # 5. Calculate mean Grad-CAM heatmap for each class
#     # Median and Max Cams dont work as well.
#     mean_class_heatmaps = {}
#     for class_label, heatmap_list in heatmaps.items():
#         mean_class_heatmaps[class_label] = np.mean(heatmap_list, axis =0)

#     # 6. plot the mean heatmaps
#     plot_MeanCam(mean_class_heatmaps, output_path, f'Mean_{cam_method.__name__}_L{classifier_target_layer}')

#     return mean_class_heatmaps


# def plot_MeanCam(mean_cam_dictionary, output_path, output_name = 'Mean Cam', plot_title = 'Mean Cam', figsize=(10,8)):
#     """
#     Plots a grid of mean arrays from a dictionary containing arrays of shape (128, 128).
#     """

#     num_classes = len(mean_cam_dictionary)
#     cols = 4  # Fixed columns
#     rows = (num_classes + cols - 1) // cols  # Compute required rows

#     fig, axes = plt.subplots(rows, cols, figsize=figsize)
#     axes = axes.flatten()

#     for idx, (title, array) in enumerate(mean_cam_dictionary.items()):
#         if idx < len(axes):  # Avoid IndexError if there are more arrays than axes
#             ax = axes[idx]
#             ax.imshow(array, cmap='viridis')
#             ax.set_title(title)
#             ax.axis("off")

#     # Turn off any remaining empty subplots
#     for idx in range(len(mean_cam_dictionary), len(axes)):
#         axes[idx].axis("off")


#     plt.tight_layout()
#     # plt.suptitle(plot_title, fontsize=18)
#     # plt.colorbar(im, ax=axes, location='right', shrink=0.7, aspect=20)  # Add a shared colorbar

#     out_name = f'{output_path}{output_name}.png'
#     plt.savefig(out_name)


def reshape_transform_for_cam(t: Tensor) -> Tensor:
    # [N,C,D,H,W] -> mean over D
    if t.dim() == 5:
        t = t.mean(dim=2)
    return t

def _group_by_class_from_dataloader(
    dataloader,
    class_names: List[str],
    max_items_per_class: Optional[int],
    device: torch.device,
) -> Dict[str, List[Tensor]]:
    buckets = {name: [] for name in class_names}
    counts = {name: 0 for name in class_names}
    cap = {name: max_items_per_class for name in class_names} if max_items_per_class else None

    for batch in dataloader:
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            raise ValueError("Dataloader must yield (inputs, targets[, ...]).")
        x, y = batch[0], batch[1]

        # normalize to [B,C,H,W]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(device)
        y = y.to(device)

        for i in range(x.size(0)):
            cls_idx = int(y[i].item())
            name = class_names[cls_idx]
            if cap and counts[name] >= cap[name]:
                continue
            # keep grad-eligible tensor (don’t detach)
            buckets[name].append(x[i])
            counts[name] += 1

        if cap and all(counts[n] >= cap[n] for n in class_names):
            break
    return buckets

def compute_class_mean_cams(
    model: torch.nn.Module,
    dataloader,
    class_names: List[str],
    device: torch.device,
    *,
    cam_method=GradCAM,
    target_layer: Optional[torch.nn.Module] = None,  # <- prefer passing a layer object
    target_layer_index: int = 3,                     # fallback: model.cnn[index]
    use_reshape_transform: bool = True,
    max_items_per_class: Optional[int] = 32,
    return_all: bool = False,
    disable_autocast: bool = True,
) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, List[np.ndarray]]]]:
    """
    Compute per-class mean CAMs. Compatible with pytorch-grad-cam >= 1.5 (no 'use_cuda' kwarg).
    """
    model.eval()
    model = model.to(device)

    # Resolve target layer
    if target_layer is None:
        try:
            target_layer = model.cnn[target_layer_index]
        except Exception as e:
            raise ValueError(
                "Provide a valid `target_layer` (module) or ensure `model.cnn[target_layer_index]` exists."
            ) from e

    cam_kwargs = {"model": model, "target_layers": [target_layer]}
    if use_reshape_transform:
        cam_kwargs["reshape_transform"] = reshape_transform_for_cam

    # IMPORTANT: Do NOT pass 'use_cuda' — removed in newer versions.
    cam = cam_method(**cam_kwargs)

    # Collect a few samples per class
    buckets = _group_by_class_from_dataloader(
        dataloader, class_names, max_items_per_class, device
    )

    all_cams: Dict[str, List[np.ndarray]] = {name: [] for name in class_names}

    # Ensure grads are enabled; keep fp32 unless you know AMP works for CAM
    torch.set_grad_enabled(True)
    autocast_ctx = (
        torch.cuda.amp.autocast(enabled=False)
        if disable_autocast and device.type == "cuda"
        else torch.cuda.amp.autocast(enabled=False)
    )

    with autocast_ctx:
        for cls_idx, cls_name in enumerate(class_names):
            for img in buckets[cls_name]:
                if img.dim() == 3:        # [C,H,W] -> [1,C,H,W]
                    img_b = img.unsqueeze(0).float()
                elif img.dim() == 4:      # [B,C,H,W] -> take first
                    img_b = img[:1].float()
                else:
                    raise ValueError(f"Unexpected input shape for CAM: {tuple(img.shape)}")
                # Make sure tensors live on same device as model
                img_b = img_b.to(device)

                model.zero_grad(set_to_none=True)
                heatmap = cam(input_tensor=img_b, targets=[ClassifierOutputTarget(cls_idx)])
                all_cams[cls_name].append(heatmap[0])  # [H,W] float np array in [0,1]

    # Aggregate means (fallback to zeros if a class bucket is empty)
    fallback = next((m for v in all_cams.values() for m in v), None)
    if fallback is None:
        raise RuntimeError("No CAMs were computed; check dataloader contents and class_names mapping.")

    mean_cams: Dict[str, np.ndarray] = {}
    for name, maps in all_cams.items():
        if not maps:
            mean_cams[name] = np.zeros_like(fallback, dtype=np.float32)
        else:
            mean_cams[name] = np.mean(np.stack(maps, axis=0), axis=0)

    return (mean_cams, all_cams) if return_all else (mean_cams, None)



def plot_mean_cams(
    mean_cams: Dict[str, np.ndarray],
    cols: int = 4,
    figsize=(10, 8),
    cmap: str = "viridis",
    suptitle: Optional[str] = None,
    add_colorbar: bool = False,
    save_path: Optional[str] = None,
    dpi: int = 200,
    tight: bool = True,
):
    """
    Plot a grid of per-class mean CAM arrays (HxW).

    Parameters
    ----------
    mean_cams : Dict[str, np.ndarray]
        Mapping from class name -> HxW CAM (float).
    cols : int
        Number of columns in the grid.
    cmap : str
        Matplotlib colormap name.
    add_colorbar : bool
        If True, add a single shared colorbar.
    save_path : Optional[str]
        If provided, saves the figure (e.g., ".../Mean_GradCAM_L3.png").
    """
    names = list(mean_cams.keys())
    n = len(names)
    rows = (n + cols - 1) // cols

    # Use a shared normalization so colors are comparable across classes
    vmin = min(float(np.min(arr)) for arr in mean_cams.values())
    vmax = max(float(np.max(arr)) for arr in mean_cams.values())
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    mappables = []
    for i, name in enumerate(names):
        ax = axes[i]
        im = ax.imshow(mean_cams[name], cmap=cmap, norm=norm)
        ax.set_title(str(name))
        ax.axis("off")
        mappables.append(im)

    # Turn off unused axes
    for j in range(n, rows * cols):
        axes[j].axis("off")

    if suptitle:
        fig.suptitle(suptitle)

    if add_colorbar and mappables:
        # Shared colorbar using the last image mappable
        fig.colorbar(mappables[-1], ax=axes[:n], fraction=0.025, pad=0.02)

    if tight:
        plt.tight_layout()

    save_path = os.path.join(save_path, "Mean_GradCAM_L3.png")
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    # return fig

def plot_InputTensors(data_dict, output_path, output_name, plot_title = 'Raw Mean Signal of TestSet',  figsize=(10, 4)):
    """
    Plots a grid of mean arrays from a dictionary containing arrays of shape (x, 5, 1, 128, 128).
    Uses only the first image of a image-sequence
    """

    num_classes = len(data_dict)
    rows = 2
    cols = (num_classes + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for idx, (title, array) in enumerate(data_dict.items()):
        if idx < len(axes):
            print('shape array',array.shape)
            if len(array.shape) == 4:
                mean_across_x = array[:, 0, :, :].mean(dim=0).cpu().numpy()
            else:
                mean_across_x = array[:, 0, 0, :, :].mean(dim=0).cpu().numpy()
            ax = axes[idx]
            im = ax.imshow(mean_across_x, cmap='viridis')
            ax.set_title(title)
            ax.axis("off")

    plt.tight_layout()
    # plt.suptitle(plot_title, fontsize=18)
    out_name = f'{output_path}/{output_name}.png'
    plt.savefig(out_name)



def plot_metric_distribution(score_dict, output_path, title):
    """
    Plots a boxplot of distribution of scores (cosine, euclidean, silhouette).
    """
    data = []
    for label, scores in score_dict.items():
        data.extend([(label, s) for s in scores])
    df = pd.DataFrame(data, columns=["Class", "Score"])

    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="Class", y="Score")
    plt.title(title)
    plt.ylabel("Score")
    plt.xlabel("Class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_similarity_matrix(matrix, class_labels, output_path, title):
    """
    Plots a heatmap for inter-class similarity/distance matrix.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, xticklabels=class_labels, yticklabels=class_labels, cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_centroid_vectors(): 
    return ''

def plot_model_stats(ax, rpt_ctrl, rpt_best, class_names,
                     rpt_all_runs=None,
                     show_legend=False, use_class_symbols=False, styles=None):

    f1_ctrl   = [rpt_ctrl[name]['f1-score']  for name in class_names]
    prec_ctrl = [rpt_ctrl[name]['precision'] for name in class_names]
    rec_ctrl  = [rpt_ctrl[name]['recall']    for name in class_names]

    if rpt_all_runs:
        f1_best   = [np.mean([rpt[name]['f1-score']  for rpt in rpt_all_runs]) for name in class_names]
        prec_best = [np.mean([rpt[name]['precision'] for rpt in rpt_all_runs]) for name in class_names]
        rec_best  = [np.mean([rpt[name]['recall']    for rpt in rpt_all_runs]) for name in class_names]
    else:
        f1_best   = [rpt_best[name]['f1-score']  for name in class_names]
        prec_best = [rpt_best[name]['precision'] for name in class_names]
        rec_best  = [rpt_best[name]['recall']    for name in class_names]

    mean_f1_ctrl   = np.mean(f1_ctrl)
    mean_f1_best   = np.mean(f1_best)
    mean_prec_ctrl = np.mean(prec_ctrl)
    mean_prec_best = np.mean(prec_best)
    mean_rec_ctrl  = np.mean(rec_ctrl)
    mean_rec_best  = np.mean(rec_best)

    x = np.arange(3)
    width = 0.35

    ax.bar(x - width/2, [mean_f1_ctrl, mean_prec_ctrl, mean_rec_ctrl], width,
           edgecolor='#b7bec4', facecolor='none', linewidth=2, label='Control')
    ax.bar(x + width/2, [mean_f1_best, mean_prec_best, mean_rec_best], width,
           edgecolor='#094c80', facecolor='none', linewidth=2, label='Model')

    scatter_offset = 0.2

    scatter_data = [
        (x[0] - width/2, f1_ctrl,   '#b7bec4'),
        (x[0] + width/2, f1_best,   '#094c80'),
        (x[1] - width/2, prec_ctrl, '#b7bec4'),
        (x[1] + width/2, prec_best, '#094c80'),
        (x[2] - width/2, rec_ctrl,  '#b7bec4'),
        (x[2] + width/2, rec_best,  '#094c80'),
    ]

    for x_bar, values, default_color in scatter_data:
        for i, class_name in enumerate(class_names):
            x_scatter = x_bar + (i - len(class_names)/2 + 0.5) * scatter_offset / len(class_names)
            y_scatter = values[i]
            if use_class_symbols and styles:
                style = get_class_style(class_name, styles)
                if style:
                    ax.scatter(x_scatter, y_scatter,
                               marker=style.get('shape', 'o'),
                               s=50,
                               facecolor=style.get('color', 'gray'),
                               edgecolor=style.get('edgecolor', 'black'),
                               linewidth=1, alpha=0.7, zorder=5)
                else:
                    ax.scatter(x_scatter, y_scatter, s=30, color=default_color, alpha=0.7, zorder=5)
            else:
                ax.scatter(x_scatter, y_scatter, s=30, color=default_color, alpha=0.7, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(['F1', 'Precision', 'Recall'], fontsize=15)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_ylim(0, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if show_legend:
        from matplotlib.patches import Rectangle
        ctrl_handle = Rectangle((0, 0), 1, 1, fill=False, edgecolor='#b7bec4', linewidth=2)
        model_handle = Rectangle((0, 0), 1, 1, fill=False, edgecolor='#094c80', linewidth=2)
        ax.legend([ctrl_handle, model_handle], ['Control', 'Model'], loc='upper right')

def plot_f1_comparison(ax, rpt_ctrl, rpt_best, class_names, show_legend=False, use_class_symbols=False, styles=None):
    """
    Plot grouped bar chart of F1-scores for control vs best.
    """
    x = np.arange(len(class_names))
    width = 0.35
    
    # Extract F1 scores
    f1_ctrl = [rpt_ctrl[name]['f1-score'] for name in class_names]
    f1_best = [rpt_best[name]['f1-score'] for name in class_names]
    
    # Plot bars
    rects1 = ax.bar(x - width/2, f1_ctrl, width, label='Control', color='#b7bec4')
    rects2 = ax.bar(x + width/2, f1_best, width, label='Best Run', color='#094c80')
    
    # Add labels and title
    ax.set_ylabel('F1-score', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    
    # Handle class labels - either text or symbols
    if use_class_symbols and styles:
        # Add symbols for each class
        for i, class_name in enumerate(class_names):
            style = get_class_style(class_name, styles)
            if style:
                ax.plot(
                    i, -0.1,
                    marker=style.get('shape', 'o'),
                    markersize=12,
                    markerfacecolor=style.get('color', 'gray'),
                    markeredgecolor=style.get('edgecolor', 'black'),
                    transform=ax.get_xaxis_transform(),
                    clip_on=False
                )
        ax.set_xticklabels([''] * len(class_names))  # Clear text labels
    else:
        # Use text labels
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    
    # Add legend if needed
    if show_legend:
        ax.legend()

    ax.set_ylim(0, 1)
    
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


# Common figure utilities
def setup_figure_matplotlib():
    """Set up common matplotlib parameters for figure generation."""
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)


def create_shared_legend(fig, handles, labels, position=(0.66, 0.84, 0.3, 0.1), 
                        ncol=None, fontsize=13):
    """
    Create a shared legend for the figure.
    
    Args:
        fig: Matplotlib figure
        handles: Legend handles
        labels: Legend labels
        position: (left, bottom, width, height) in figure coordinates
        ncol: Number of columns (if None, uses len(labels))
        fontsize: Font size for legend
    """
    if ncol is None:
        ncol = len(labels)
    
    # Create a new subplot for the legend
    legend_ax = fig.add_axes(position)
    legend_ax.axis('off')
    
    # Add legend to the new subplot
    legend = legend_ax.legend(handles, labels, 
                             loc='center', 
                             ncol=ncol,
                             fontsize=fontsize, 
                             frameon=False)
    legend.set_in_layout(False)
    return legend


def add_l_shaped_axis(ax, axis_length=20.0, show_labels=True):
    """
    Add L-shaped corner axis to the plot with consistent length.
    
    Args:
        ax: Matplotlib axis to modify
        axis_length: Length of the L-shape in data coordinates
        show_labels: Whether to show the axis labels
    """
    # Get current axis limits from the plot
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    # Calculate the offset for labels (5% of the axis range)
    x_offset = (xmax - xmin) * 0.05
    y_offset = (ymax - ymin) * 0.05
    
    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add L-shaped corner axis with consistent length
    # Horizontal line
    ax.axhline(y=ymin, xmin=0, xmax=axis_length/(xmax-xmin), 
               color='black', linewidth=1, clip_on=False)
    # Vertical line
    ax.axvline(x=xmin, ymin=0, ymax=axis_length/(ymax-ymin), 
               color='black', linewidth=1, clip_on=False)
    
    # Add axis labels at the ends of the L if show_labels is True
    if show_labels:
        ax.text(xmin, ymin - y_offset, 't-SNE 1', 
                ha='left', va='top', fontsize=10)
        ax.text(xmin - x_offset, ymin, 't-SNE 2', 
                ha='right', va='bottom', fontsize=10, rotation=90)
    
    # Hide default ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Make sure the axis limits stay the same
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def get_group_map(task, class_names):
    """
    Return the group mapping for the given task and class names.
    
    Args:
        task: Task name
        class_names: List of class names
        
    Returns:
        Dictionary mapping group names to class indices
    """
    if task == 'MetabolicState_2':
        return {
            'S': [0], 
            'F': [1]
        }
    
    if task == 'State_Modality_6':
        return {
            'S': [0, 2, 4], 
            'F': [1, 3, 5], 
            'O': [0, 1], 
            'T': [2, 3], 
            'O/T': [4, 5]
        }
    
    if task == 'State_Modality_Valence_16':
        # Build Pos/Neg based on name matching
        pos_inds = [i for i, n in enumerate(class_names) if '+' in n and '-' not in n]
        neg_inds = [i for i, n in enumerate(class_names) if '-' in n and '+' not in n]
        return {
            'S': [0, 1, 4, 5, 8, 9, 10, 11],
            'F': [2, 3, 6, 7, 12, 13, 14, 15],
            'O': [0, 1, 2, 3],
            'T': [4, 5, 6, 7],
            'O/T': [8, 9, 10, 11, 12, 13],
            '+': pos_inds,
            '-': neg_inds,
            '+/-': [10, 11, 14, 15]
        }
    
    return {}

        
def plot_precision_recall_comparison(ax, rpt_best, class_names, show_legend=False, use_class_symbols=False, styles=None):
    """
    Plot grouped bar chart of best-run precision and recall.
    """
    n_classes = len(class_names)
    width = 0.35
    ind = np.arange(n_classes)
    
    acc_best = rpt_best['accuracy']
    prec_best = [rpt_best[name]['precision'] for name in class_names]
    rec_best = [rpt_best[name]['recall'] for name in class_names]
    
    ax.bar(ind - width/2, prec_best, width, label='Precision', color='royalblue')
    ax.bar(ind + width/2, rec_best, width, label='Recall', color='cornflowerblue')
    ax.axhline(acc_best, ax.get_xlim()[0], ax.get_xlim()[1], color='gray', linestyle='--')
    
    # Handle class labels - either text or symbols
    if use_class_symbols and styles:
        ax.set_xticklabels([''] * len(class_names))  # Clear text labels
        # Add symbols for each class
        for i, class_name in enumerate(class_names):
            style = get_class_style(class_name, styles)
            if style:
                _plot_class_symbol(ax, i, -0.1, style, markersize=12, clip_on=False)
    else:
        # Use text labels
        ax.set_xticklabels(class_names, rotation=90, ha='right')
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    
    if show_legend:
        ax.legend()
    
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


# Common figure utilities
def setup_figure_matplotlib():
    """Set up common matplotlib parameters for figure generation."""
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)


def create_shared_legend(fig, handles, labels, position=(0.66, 0.84, 0.3, 0.1), 
                        ncol=None, fontsize=13):
    """
    Create a shared legend for the figure.
    
    Args:
        fig: Matplotlib figure
        handles: Legend handles
        labels: Legend labels
        position: (left, bottom, width, height) in figure coordinates
        ncol: Number of columns (if None, uses len(labels))
        fontsize: Font size for legend
    """
    if ncol is None:
        ncol = len(labels)
    
    # Create a new subplot for the legend
    legend_ax = fig.add_axes(position)
    legend_ax.axis('off')
    
    # Add legend to the new subplot
    legend = legend_ax.legend(handles, labels, 
                             loc='center', 
                             ncol=ncol,
                             fontsize=fontsize, 
                             frameon=False)
    legend.set_in_layout(False)
    return legend


def add_l_shaped_axis(ax, axis_length=20.0, show_labels=True):
    """
    Add L-shaped corner axis to the plot with consistent length.
    
    Args:
        ax: Matplotlib axis to modify
        axis_length: Length of the L-shape in data coordinates
        show_labels: Whether to show the axis labels
    """
    # Get current axis limits from the plot
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    # Calculate the offset for labels (5% of the axis range)
    x_offset = (xmax - xmin) * 0.05
    y_offset = (ymax - ymin) * 0.05
    
    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add L-shaped corner axis with consistent length
    # Horizontal line
    ax.axhline(y=ymin, xmin=0, xmax=axis_length/(xmax-xmin), 
               color='black', linewidth=1, clip_on=False)
    # Vertical line
    ax.axvline(x=xmin, ymin=0, ymax=axis_length/(ymax-ymin), 
               color='black', linewidth=1, clip_on=False)
    
    # Add axis labels at the ends of the L if show_labels is True
    if show_labels:
        ax.text(xmin, ymin - y_offset, 't-SNE 1', 
                ha='left', va='top', fontsize=10)
        ax.text(xmin - x_offset, ymin, 't-SNE 2', 
                ha='right', va='bottom', fontsize=10, rotation=90)
    
    # Hide default ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Make sure the axis limits stay the same
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def get_group_map(task, class_names):
    """
    Return the group mapping for the given task and class names.
    
    Args:
        task: Task name
        class_names: List of class names
        
    Returns:
        Dictionary mapping group names to class indices
    """
    if task == 'MetabolicState_2':
        return {
            'S': [0], 
            'F': [1]
        }
    
    if task == 'State_Modality_6':
        return {
            'S': [0, 2, 4], 
            'F': [1, 3, 5], 
            'O': [0, 1], 
            'T': [2, 3], 
            'O/T': [4, 5]
        }
    
    if task == 'State_Modality_Valence_16':
        # Build Pos/Neg based on name matching
        pos_inds = [i for i, n in enumerate(class_names) if '+' in n and '-' not in n]
        neg_inds = [i for i, n in enumerate(class_names) if '-' in n and '+' not in n]
        return {
            'S': [0, 1, 4, 5, 8, 9, 10, 11],
            'F': [2, 3, 6, 7, 12, 13, 14, 15],
            'O': [0, 1, 2, 3],
            'T': [4, 5, 6, 7],
            'O/T': [8, 9, 10, 11, 12, 13],
            '+': pos_inds,
            '-': neg_inds,
            '+/-': [10, 11, 14, 15]
        }
    
    return {}