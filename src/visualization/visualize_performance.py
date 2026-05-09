import os
import textwrap
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
from src.visualization.figure_base import FONT_SIZES
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.manifold import TSNE

from src.utils.helpers import (
    HALF_CIRCLE_LEFT, HALF_CIRCLE_RIGHT,
    scatter_bicolor, scatter_bicolor_cloud,
    _color_for_group, get_group_map,
)

Tensor = torch.Tensor

# def get_class_style(class_name, styles):
#     """Get style properties for a given class name.
    
#     Args:
#         class_name: The name of the class to get style for
#         styles: Dictionary of style properties from styles.yaml
        
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
        styles: Dictionary of style properties from styles.yaml
        
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


def build_class_styles(colors, edgecolors, shapes, bicolor_info=None):
    """
    Build a list of per-class style dicts from the parallel arrays returned
    by get_style().  This avoids the fragile name-based lookup in
    get_class_style and guarantees that bicolor entries are included.

    Returns:
        List[dict] — one style dict per class, in class-index order.
    """
    if bicolor_info is None:
        bicolor_info = {}
    class_styles = []
    for i in range(len(colors)):
        s = {
            'color': colors[i],
            'edgecolor': edgecolors[i],
            'shape': shapes[i] if i < len(shapes) else 'o',
        }
        if i in bicolor_info:
            s['bicolor'] = True
            s.update(bicolor_info[i])
        class_styles.append(s)
    return class_styles


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
                        axis_labeling='both', use_class_symbols=False, styles=None, class_styles=None):
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
        fontsize_annot = FONT_SIZES['heatmap_cell']
        fontsize_ticks = FONT_SIZES['tick']
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
    if use_class_symbols and (styles or class_styles):
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
    if use_class_symbols and (styles or class_styles):
        # Get the style for each class
        for i, class_name in enumerate(class_names):
            style = (class_styles[i] if class_styles else None) or get_class_style(class_name, styles)
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


def plot_model_stats(ax, rpt_ctrl, rpt_best, class_names,
                     rpt_all_runs=None,
                     show_legend=False, use_class_symbols=False, styles=None,
                     class_styles=None):

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
            if use_class_symbols and (styles or class_styles):
                style = (class_styles[i] if class_styles else None) or get_class_style(class_name, styles)
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
    ax.set_xticklabels(['F1', 'Precision', 'Recall'], fontsize=FONT_SIZES['label'])
    ax.set_ylabel('Accuracy', fontsize=FONT_SIZES['label'])
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
                ha='left', va='top', fontsize=FONT_SIZES['label'])
        ax.text(xmin - x_offset, ymin, 't-SNE 2',
                ha='right', va='bottom', fontsize=FONT_SIZES['label'], rotation=90)
    
    # Hide default ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Make sure the axis limits stay the same
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)




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


# ═══════════════════════════════════════════════════════════════════
# Functions moved from run_figure_hLatent — latent-space visualisation
# ═══════════════════════════════════════════════════════════════════

def _short_name(n: str) -> str:
    """Shorten LaTeX class names for plot labels."""
    return n.replace('$^{+}$', '+').replace('$^{-}$', '-').replace(' (S)', ' S').replace(' (F)', ' F')


def _compute_biological_axes(X, labels, task_key, class_names):
    """
    Compute biological axes and project class centroids.
    Returns: (projections_dict, axes_dict, displacements)
    """
    global_mu = X.mean(axis=0)
    unique_labels = np.sort(np.unique(labels))
    centroids = np.array([X[labels == l].mean(axis=0) for l in unique_labels])
    displacements = centroids - global_mu

    if task_key == 'MetabolicState_2':
        mean_S = X[labels == 0].mean(axis=0)
        mean_F = X[labels == 1].mean(axis=0)
        state_axis = mean_F - mean_S
        state_axis /= np.linalg.norm(state_axis)
        proj_state = displacements @ state_axis
        return {'state': proj_state}, {'state': state_axis}, displacements

    if task_key == 'State_Modality_6':
        mean_S = X[np.isin(labels, [0, 2, 4])].mean(axis=0)
        mean_F = X[np.isin(labels, [1, 3, 5])].mean(axis=0)
        mean_O = X[np.isin(labels, [0, 1])].mean(axis=0)
        mean_T = X[np.isin(labels, [2, 3])].mean(axis=0)

        state_axis = mean_F - mean_S
        state_axis /= np.linalg.norm(state_axis)
        modality_axis = mean_T - mean_O
        modality_axis /= np.linalg.norm(modality_axis)

        proj_state = displacements @ state_axis
        proj_modality = displacements @ modality_axis

        return {'state': proj_state, 'modality': proj_modality}, \
               {'state': state_axis, 'modality': modality_axis}, displacements

    if task_key == 'State_Modality_Valence_16':
        mean_S = X[np.isin(labels, [0, 1, 4, 5, 8, 9, 10, 11])].mean(axis=0)
        mean_F = X[np.isin(labels, [2, 3, 6, 7, 12, 13, 14, 15])].mean(axis=0)
        mean_O = X[np.isin(labels, [0, 1, 2, 3])].mean(axis=0)
        mean_T = X[np.isin(labels, [4, 5, 6, 7])].mean(axis=0)

        pos_inds = [i for i, n in enumerate(class_names) if '$^{+}$' in n and '$^{-}$' not in n]
        neg_inds = [i for i, n in enumerate(class_names) if '$^{-}$' in n and '$^{+}$' not in n]
        mean_pos = X[np.isin(labels, pos_inds)].mean(axis=0)
        mean_neg = X[np.isin(labels, neg_inds)].mean(axis=0)

        state_axis = mean_F - mean_S
        state_axis /= np.linalg.norm(state_axis)
        modality_axis = mean_T - mean_O
        modality_axis /= np.linalg.norm(modality_axis)
        valence_axis = mean_pos - mean_neg
        valence_axis /= np.linalg.norm(valence_axis)

        proj_state = displacements @ state_axis
        proj_modality = displacements @ modality_axis
        proj_valence = displacements @ valence_axis

        return {'state': proj_state, 'modality': proj_modality, 'valence': proj_valence}, \
               {'state': state_axis, 'modality': modality_axis, 'valence': valence_axis}, displacements


def _clean_axis(ax):
    """Remove all spines, ticks, and labels — blank canvas."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def _add_axis_indicator(ax, axis_names, fontsize=None):
    """
    Add a clean L-shaped axis indicator in the bottom-left corner.
    All lines meet at a single origin point. Uses axes-fraction transform.
    
    axis_names: list of axis name strings
      - 1 name  → horizontal line only (e.g. ['State'])
      - 2 names → L-shape: horizontal + vertical (e.g. ['State', 'Modality'])
      - 3 names → L-shape + diagonal for 3rd axis
    """
    # Origin and arm length in axes fraction
    if fontsize is None:
        fontsize = FONT_SIZES['label']
    x0, y0 = 0.06, 0.06
    length = 0.13
    trans = ax.transAxes

    # Horizontal arm
    if len(axis_names) >= 1:
        ax.plot([x0, x0 + length], [y0, y0], '-', color='black', lw=1.0,
                transform=trans, clip_on=False)
        ax.text(x0 + length / 2, y0 - 0.035, axis_names[0],
                transform=trans, ha='center', va='top', fontsize=fontsize)

    # Vertical arm (connected at same origin)
    if len(axis_names) >= 2:
        ax.plot([x0, x0], [y0, y0 + length], '-', color='black', lw=1.0,
                transform=trans, clip_on=False)
        ax.text(x0 - 0.02, y0 + length / 2, axis_names[1],
                transform=trans, ha='right', va='center', fontsize=fontsize,
                rotation=90)

    # Diagonal arm (connected at same origin)
    if len(axis_names) >= 3:
        diag = length * 0.75
        dx = diag * np.cos(np.deg2rad(45))
        dy = diag * np.sin(np.deg2rad(45))
        ax.plot([x0, x0 + dx], [y0, y0 + dy], '-', color='black', lw=1.0,
                transform=trans, clip_on=False)
        ax.text(x0 + dx + 0.015, y0 + dy + 0.01, axis_names[2],
                transform=trans, ha='left', va='bottom', fontsize=fontsize)




def plot_1d_marginal(
    ax: plt.Axes,
    proj: np.ndarray,
    class_names: List[str],
    colors,
    edges,
    bicolor_info: Dict = None,
    axis_name: str = '',
    s: int = 180,
) -> None:
    """Plot class centroids as 1D markers along a single biological axis.

    Horizontal axis = projection value; all points placed at y=0.
    Marker conventions mirror plot_biological_axes_panel (open = Starved, filled = Fed,
    split bicolor = Conflicting).
    """
    if bicolor_info is None:
        bicolor_info = {}

    _clean_axis(ax)

    xlim = float(np.abs(proj).max()) * 1.4
    if xlim == 0:
        xlim = 1.0

    ax.plot([-xlim, xlim], [0, 0], '-', color='gray', lw=1.2, alpha=0.6, zorder=1)
    ax.plot(0, 0, '+', color='gray', ms=8, mew=1.5, zorder=2)

    for i, cname in enumerate(class_names):
        is_starved = '(S)' in cname or cname == 'Starved'
        lw = 2.0 if is_starved else 0.5
        x = float(proj[i])
        if i in bicolor_info:
            scatter_bicolor(ax, x, 0.0, bicolor_info[i], s=s, linewidth=lw, zorder=3)
        else:
            ax.scatter(x, 0.0, c=colors[i], edgecolors=edges[i],
                       linewidth=lw, s=s, zorder=3)

    ax.set_xlim(-xlim * 2.0, xlim * 2.0)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel(axis_name, fontsize=FONT_SIZES['label'])


def plot_biological_axes_panel(
    ax,
    X: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    task: str,
    colors: Dict,
    edges: Dict,
    bicolor_info: Dict = None,
    fig: plt.Figure = None,
    gs: 'GridSpec' = None,
    row: int = 0,
) -> None:
    """
    Plot biological axes projections with cosine similarity annotations.
    Clean style: plain gray lines, text endpoint labels, no axis frames,
    no class name annotations, cosine + variance box in top-left.
    
    All three rows use 2D axes (16-class uses oblique projection).
    """
    if bicolor_info is None:
        bicolor_info = {}
    projections, axes_dict, displacements = _compute_biological_axes(X, labels, task, class_names)
    n_classes = len(np.unique(labels))

    # Variance explained
    total_var = np.sum(displacements ** 2)
    explained = sum(np.sum(p ** 2) for p in projections.values())
    pct = explained / total_var * 100

    # ── 2-class: 1D number line ──
    if task == 'MetabolicState_2':
        proj_s = projections['state']

        # Cosine similarity between the 2 displacement vectors
        cos_val = np.dot(displacements[0], displacements[1]) / \
                  (np.linalg.norm(displacements[0]) * np.linalg.norm(displacements[1]))

        _clean_axis(ax)

        # Gray line
        xlim = np.abs(proj_s).max() * 1.4
        ax.plot([-xlim, xlim], [0, 0], '-', color='gray', lw=1.2, alpha=0.6, zorder=1)

        # Origin cross
        ax.plot(0, 0, '+', color='gray', ms=8, mew=1.5, zorder=2)

        # Centroids (no labels)
        for i in range(n_classes):
            is_starved = '(S)' in class_names[i] or class_names[i] == 'Starved'
            lw = 2.0 if is_starved else 0.5
            if i in bicolor_info:
                scatter_bicolor(ax, proj_s[i], 0, bicolor_info[i], s=200, linewidth=lw, zorder=3)
            else:
                ax.scatter(proj_s[i], 0, c=colors[i], edgecolors=edges[i],
                           linewidth=lw, s=200, zorder=3)

        # Endpoint text labels — placed beyond the line ends for clearance
        ax.text(-xlim * 1.15, 0, 'Starved', ha='right', va='center', fontsize=FONT_SIZES['annotation'])
        ax.text(xlim * 1.15, 0, 'Fed', ha='left', va='center', fontsize=FONT_SIZES['annotation'])

        ax.set_xlim(-xlim * 2.2, xlim * 2.2)
        ax.set_ylim(-0.5, 0.5)

        # Axis indicator
        _add_axis_indicator(ax, ['State'])

    # ── 6-class: 2D scatter ──
    elif task == 'State_Modality_6':
        proj_s, proj_m = projections['state'], projections['modality']

        # Cosine similarity between axes
        cos_sm = np.dot(axes_dict['state'], axes_dict['modality'])

        _clean_axis(ax)

        # Axis limits
        all_vals = np.concatenate([proj_s, proj_m])
        lim = np.abs(all_vals).max() * 1.4

        # Plain gray crossing lines
        ax.plot([-lim, lim], [0, 0], '-', color='gray', lw=1.0, alpha=0.5, zorder=1)
        ax.plot([0, 0], [-lim, lim], '-', color='gray', lw=1.0, alpha=0.5, zorder=1)

        # Origin cross
        ax.plot(0, 0, '+', color='gray', ms=8, mew=1.5, zorder=2)

        # Centroids (no labels)
        for i in range(n_classes):
            is_starved = '(S)' in class_names[i]
            lw = 2.0 if is_starved else 0.5
            if i in bicolor_info:
                scatter_bicolor(ax, proj_s[i], proj_m[i], bicolor_info[i], s=180, linewidth=lw, zorder=3)
            else:
                ax.scatter(proj_s[i], proj_m[i], c=colors[i], edgecolors=edges[i],
                           linewidth=lw, s=180, zorder=3)

        # Endpoint text labels
        ax.text(-lim, 0, 'Starved  ', ha='right', va='center', fontsize=FONT_SIZES['annotation'])
        ax.text(lim, 0, '  Fed', ha='left', va='center', fontsize=FONT_SIZES['annotation'])
        ax.text(0, lim, 'Taste', ha='center', va='bottom', fontsize=FONT_SIZES['annotation'])
        ax.text(0, -lim, 'Odor', ha='center', va='top', fontsize=FONT_SIZES['annotation'])

        ax.set_xlim(-lim * 1.7, lim * 1.7)
        ax.set_ylim(-lim * 1.5, lim * 1.5)

        # Axis indicator
        _add_axis_indicator(ax, ['State', 'Modality'])

    # ── 16-class: flat 2D with oblique 3rd axis ──
    elif task == 'State_Modality_Valence_16':
        proj_s = projections['state']
        proj_m = projections['modality']
        proj_v = projections['valence']

        # Pairwise cosine similarities
        cos_sm = np.dot(axes_dict['state'], axes_dict['modality'])
        cos_sv = np.dot(axes_dict['state'], axes_dict['valence'])
        cos_mv = np.dot(axes_dict['modality'], axes_dict['valence'])

        _clean_axis(ax)

        # Oblique projection: state → x, modality → y, valence → diagonal
        angle = np.deg2rad(35)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Scale valence contribution so it doesn't dominate
        v_scale = 0.6
        x_pts = proj_s + proj_v * v_scale * cos_a
        y_pts = proj_m + proj_v * v_scale * sin_a

        # Axis limits
        all_vals = np.concatenate([x_pts, y_pts])
        lim = np.abs(all_vals).max() * 1.3

        # Draw three gray axis lines through origin
        ax.plot([-lim, lim], [0, 0], '-', color='gray', lw=1.0, alpha=0.5, zorder=1)
        ax.plot([0, 0], [-lim, lim], '-', color='gray', lw=1.0, alpha=0.5, zorder=1)
        diag_len = lim * 0.85
        ax.plot([-diag_len * cos_a, diag_len * cos_a],
                [-diag_len * sin_a, diag_len * sin_a],
                '-', color='gray', lw=1.0, alpha=0.5, zorder=1)

        # Origin cross
        ax.plot(0, 0, '+', color='gray', ms=8, mew=1.5, zorder=2)

        # Centroids (no labels)
        for i in range(n_classes):
            is_starved = '(S)' in class_names[i]
            lw = 2.0 if is_starved else 0.5
            if i in bicolor_info:
                scatter_bicolor(ax, x_pts[i], y_pts[i], bicolor_info[i], s=120, linewidth=lw, zorder=3)
            else:
                ax.scatter(x_pts[i], y_pts[i], c=colors[i], edgecolors=edges[i],
                           linewidth=lw, s=120, zorder=3)

        # Endpoint text labels
        ax.text(-lim, 0, 'Starved  ', ha='right', va='center', fontsize=FONT_SIZES['annotation'])
        ax.text(lim, 0, '  Fed', ha='left', va='center', fontsize=FONT_SIZES['annotation'])
        ax.text(0, lim, 'Taste', ha='center', va='bottom', fontsize=FONT_SIZES['annotation'])
        ax.text(0, -lim, 'Odor', ha='center', va='top', fontsize=FONT_SIZES['annotation'])
        ax.text(diag_len * cos_a, diag_len * sin_a, '  App.', ha='left', va='bottom', fontsize=FONT_SIZES['annotation'])
        ax.text(-diag_len * cos_a, -diag_len * sin_a, 'Avers.  ', ha='right', va='top', fontsize=FONT_SIZES['annotation'])

        ax.set_xlim(-lim * 1.7, lim * 1.7)
        ax.set_ylim(-lim * 2.0, lim * 1.6)

        # Axis indicator
        _add_axis_indicator(ax, ['State', 'Modality', 'Valence'])


def plot_centroid_vectors(
    latent: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    task: str,
    ax: plt.Axes,
    plot_labels: bool = True,
    styles: Optional[Dict[str, Any]] = None,
    fixed_lim: Optional[float] = None,
) -> None:
    """
    Plot displacement vectors from the global mean to task-defined group centroids
    in a 2D latent space (e.g., t-SNE). Re-centers so the global mean is at (0, 0).
    latent: (n_samples, 2); labels: (n_samples,)
    fixed_lim: if provided, use this as the axis limit (±fixed_lim) instead of auto-computing
    """
    assert latent.ndim == 2 and latent.shape[1] == 2, "latent must be (N, 2)"
    assert labels.shape[0] == latent.shape[0], "labels length must match latent"

    # 1) Center so that global mean is the origin
    global_mu = latent.mean(axis=0)
    Z = latent - global_mu

    # 2) Task-specific grouping
    group_map = get_group_map(task, class_names)

    # 3) Colors: unify lookup with fallback
    def _col(key: str) -> Any:
        if styles is None:
            return "k"
        d = styles.get(key, {})
        return d.get("arrow_color", d.get("color", "k"))

    # Compute centroids
    centroids = {}
    for name, inds in group_map.items():
        mask = np.isin(labels, inds)
        if np.any(mask):
            centroids[name] = Z[mask].mean(axis=0)

    # 4) Plot arrows
    for name, vec in centroids.items():
        col = (
            _col("starved")            if name == "S"   else
            _col("fed")                if name == "F"   else
            _col("odor")               if name == "O"   else
            _col("taste")              if name == "T"   else
            _col("odor_taste")         if name == "O/T" else
            _col("positive")           if name == "+"   else
            _col("negative")           if name == "-"   else
            _col("positive_negative")  if name == "+/-" else
            "k"
        )
        x0, y0 = 0.0, 0.0
        dx, dy = float(vec[0]), float(vec[1])
        ax.arrow(
            x0, y0, dx, dy,
            head_width=6, head_length=6, linewidth=2,
            alpha=0.7, color=col, length_includes_head=True, zorder=2
        )
        if plot_labels:
            mid_x, mid_y = x0 + dx / 2.0, y0 + dy / 2.0
            ax.annotate(
                name, xy=(mid_x, mid_y), xytext=(0, 8),
                textcoords="offset points", ha="center", va="center",
                fontsize=FONT_SIZES['annotation'], color=col, zorder=3
            )

    # 5) Cosmetics
    ax.axhline(0, lw=0.8, alpha=0.15, color="0.3", zorder=1)
    ax.axvline(0, lw=0.8, alpha=0.15, color="0.3", zorder=1)

    if fixed_lim is not None:
        R = fixed_lim
    elif centroids:
        R = np.max([np.hypot(*v) for v in centroids.values()])
        R = 1.1 * R if R > 0 else 1.0
    else:
        R = 1.0
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_aspect("equal", adjustable="box")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_accuracy_vs_dimension(
    ax: plt.Axes,
    task_results: Dict,
    task_name: str,
    color: str,
    row: int = 0,
    primary_family: str = "E",
    overlay_alt_family: bool = True,
    grid_dims: Optional[List[int]] = None,
    best_dim: Optional[int] = None,
    baseline: Optional[float] = None,
) -> None:
    """Plot accuracy vs latent dimension for a given task."""
    def _extract_family_stats(family_prefix: str):
        fam_items = [(k, v) for k, v in task_results.get('runs', {}).items() if k.startswith(family_prefix)]
        if not fam_items:
            return [], {}
        dim_accuracies = {}
        for dim_key, runs in fam_items:
            try:
                dim = int(dim_key[1:])
            except Exception:
                continue
            accs = []
            for run in runs:
                acc = None
                if "accuracy" in run and isinstance(run["accuracy"], (int, float)):
                    acc = float(run["accuracy"])
                else:
                    d = run.get('data', {}) or {}
                    acc = (
                        d.get('test_accuracy', None)
                        if d.get('test_accuracy', None) is not None else
                        d.get('accuracy', None)
                        if d.get('accuracy', None) is not None else
                        (d.get('metrics', {}) or {}).get('test_accuracy', None)
                    )
                    if acc is not None:
                        acc = float(acc)
                if acc is not None:
                    accs.append(acc)
            if not accs:
                print(f"[DEBUG] _extract_family_stats: Empty acc list - task={task_name}, "
                      f"family_prefix={family_prefix}, dim_key={dim_key}, n_runs={len(runs)}")
            if accs:
                accs = np.asarray(accs, dtype=float)
                dim_accuracies[dim] = {
                    'best_accuracy': float(np.max(accs) * 100.0),
                    'mean': float(np.mean(accs) * 100.0),
                    'median': float(np.median(accs) * 100.0),
                    'quantiles': np.percentile(accs, [25, 75]) * 100.0,
                    'std': float(np.std(accs, ddof=1) * 100.0) if len(accs) > 1 else 0.0,
                    'n': int(len(accs)),
                }
        dims_sorted = sorted(dim_accuracies.keys())
        return dims_sorted, dim_accuracies

    if grid_dims is None:
        grid_dims = [4, 8, 16, 32, 64]
    ix = {d: i for i, d in enumerate(grid_dims)}

    TARGET_DIM = best_dim if best_dim is not None else 16
    REQUIRE_BOTH_FAMILIES = False
    STAR_STAT = "best_accuracy"

    dims_p, stats_p = _extract_family_stats(primary_family)
    if not dims_p:
        ax.axis('off')
        return

    alt_family = "H" if primary_family == "E" else "E"
    dims_a, stats_a = _extract_family_stats(alt_family)

    dims_p = [d for d in dims_p if d in ix]
    x_p = np.array([ix[d] for d in dims_p], dtype=int)
    med_p = [stats_p[d]['median'] for d in dims_p]
    mean_p = [stats_p[d]['mean'] for d in dims_p]
    q_p = [stats_p[d]['quantiles'] for d in dims_p]
    best_p = [stats_p[d]['best_accuracy'] for d in dims_p]
    n_p = [stats_p[d]['n'] for d in dims_p]
    qlow_p = np.array([q[0] for q in q_p])
    qupp_p = np.array([q[1] for q in q_p])

    has_p = TARGET_DIM in stats_p and TARGET_DIM in ix
    has_a = TARGET_DIM in stats_a and TARGET_DIM in ix
    if has_p and (not REQUIRE_BOTH_FAMILIES or has_a):
        star_x = ix[TARGET_DIM]
        star_y = stats_p[TARGET_DIM][STAR_STAT]
    else:
        star_x = None
        star_y = None

    if overlay_alt_family and dims_a:
        dims_a = [d for d in dims_a if d in ix]
        if dims_a:
            x_a   = np.array([ix[d] for d in dims_a], dtype=int)
            med_a = [stats_a[d]['median']     for d in dims_a]
            q_a   = [stats_a[d]['quantiles']  for d in dims_a]
            qlow_a = np.array([q[0] for q in q_a])
            qupp_a = np.array([q[1] for q in q_a])

            err_low_a = np.array([med_a[i] - qlow_a[i] for i in range(len(med_a))])
            err_up_a = np.array([qupp_a[i] - med_a[i] for i in range(len(med_a))])

            ax.plot(
                x_a, med_a, 'o-', color='darkcyan', markersize=8, linewidth=1.5, alpha=0.8,
                markerfacecolor='white', markeredgecolor='darkcyan', markeredgewidth=1.5,
                label=f"{alt_family}", zorder=2
            )
            ax.fill_between(x_a, qlow_a, qupp_a, alpha=0.3, color='darkcyan', linewidth=0, zorder=1)
            ax.errorbar(x_a, med_a, yerr=[err_low_a, err_up_a], fmt='none', capsize=2,
                       elinewidth=1, alpha=0.9, zorder=3, color='darkcyan')

    err_low_p = np.array([med_p[i] - qlow_p[i] for i in range(len(med_p))])
    err_up_p = np.array([qupp_p[i] - med_p[i] for i in range(len(med_p))])

    ax.plot(x_p, med_p, 'o-', color='k', markersize=8, linewidth=1.5, alpha=0.8,
            markerfacecolor='white', markeredgecolor='k', markeredgewidth=1.5, label=f"{primary_family}", zorder=2)
    ax.fill_between(x_p, qlow_p, qupp_p, alpha=0.5, color='gray', linewidth=0, zorder=1)
    ax.errorbar(x_p, med_p, yerr=[err_low_p, err_up_p], fmt='none', capsize=2,
               elinewidth=1, alpha=0.9, zorder=3, color='k')

    if star_x is not None and star_y is not None:
        ax.plot(star_x, star_y, marker='*', markersize=10, color='gold',
                markeredgecolor='k', markeredgewidth=0.8, zorder=5)

    if task_name == 'i. State':
        chance_level, y_min, y_max = 50.0, 70, 100
    elif task_name == 'ii. State, Modality':
        chance_level, y_min, y_max = 100/6, 70, 100
    elif task_name == 'iii. State, Modality, Valence':
        chance_level, y_min, y_max = 100/16, 60, 90
    else:
        chance_level, y_min, y_max = 0.0, 60, 100

    N = len(grid_dims)
    ax.set_xticks(np.arange(N))
    ax.set_xticklabels([str(d) for d in grid_dims])
    if row == 2:
        ax.set_xlabel('Latent Dimension', fontsize=FONT_SIZES['label'])

    ax.set_ylabel('Accuracy (%)', fontsize=FONT_SIZES['label'])
    ref_level = baseline if baseline is not None else chance_level
    ref_label = f'Baseline Accuracy: {ref_level:.1f}%' if baseline is not None else f'Chance: {ref_level:.1f}%'
    ax.axhline(y=ref_level, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.text(0.35, y_min + 1, ref_label,
            transform=ax.get_yaxis_transform(), color='gray', va='bottom', fontsize=FONT_SIZES['label'])

    ax.spines['left'].set_bounds(y_min, y_max)
    ax.spines['bottom'].set_bounds(0, max(0, N-1))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    y_ticks = np.arange(np.ceil(ref_level/10)*10, min(101, np.floor(y_max/10)*10 + 1), 10, dtype=int)
    ax.yaxis.set_ticks(y_ticks)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.5, N - 0.5)

    if row < 2:
        ax.set_xticklabels([]); ax.spines['bottom'].set_visible(False); ax.tick_params(axis='x', which='both', length=0)
    else:
        ax.spines['bottom'].set_position(('outward', 10)); ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)

    if overlay_alt_family:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, loc="upper right", fontsize=FONT_SIZES['legend'], frameon=False)


def plot_tsne_panel(
    fig: plt.Figure,
    gs: GridSpec,
    row: int,
    col: int,
    run_dict: Dict,
    class_names: List[str],
    colors: Dict,
    edges: Dict,
    shapes: Dict,
    styles: Dict = None,
    bicolor_info: Dict = None,
    use_l_axis: bool = False,
    is_control: bool = False,
    plot_centroids: bool = False,
    task: str = None,
    logger=None,
    centroid_fixed_lim: Optional[float] = None,
) -> None:
    """Plot t-SNE visualization with optional centroids.
    
    Args:
        fig: Figure to plot on
        gs: GridSpec for layout
        row: Grid row
        col: Grid column
        run_dict: Data for plotting
        class_names: List of class names
        colors: Color mapping
        edges: Edge colors
        shapes: Marker shapes
        styles: Style dictionary for centroids
        bicolor_info: Dict mapping class index to bicolor style info
        use_l_axis: Use L-shaped axis
        is_control: If control plot
        plot_centroids: Add centroid vectors
        task: Task name for centroids
        centroid_fixed_lim: Fixed axis limit for centroid plots (±lim)
    """
    # Get data
    data_key = 'control' if is_control else \
              next((k for k in ['best', 'control'] if k in run_dict),
                  next(k for k in run_dict if k != '__class_names__'))
    if logger:
        logger.debug(f"plot_tsne_panel: key='{data_key}', task='{task}' (row={row}, col={col})")
    data = run_dict[data_key]

    # Create subplot
    ax = fig.add_subplot(gs[row, col])

    # Only plot t-SNE if not in centroids column
    if not plot_centroids:
        plot_tsne_latent(
            latent_2d=data['tsne_2d'],
            labels_np=data['latent_labels'],
            class_names=class_names,
            ax=ax,
            xlim=[-115, 125] if not use_l_axis and not is_control else None,
            ylim=[-115, 125] if not use_l_axis and not is_control else None,
            colors=colors,
            shapes=shapes,
            edgecolors=edges,
            bicolor_info=bicolor_info,
            legend=False,
            draw_axis=not use_l_axis,
            draw_title=False
        )
    # Add centroids if requested
    if plot_centroids and styles and task:
        plot_centroid_vectors(
            latent=data['tsne_2d'],
            labels=data['latent_labels'],
            class_names=class_names,
            task=task,
            ax=ax,
            plot_labels=False,
            styles=styles,
            fixed_lim=centroid_fixed_lim,
        )
        ax.plot(0, 0, 'ko', markersize=3, zorder=10)

    # Style axes
    if use_l_axis:
        add_l_shaped_axis(ax, show_labels=(is_control and row == 0))
    if is_control:
        ax.set_ylabel('')


def draw_legend_panel(fig, styles, line_y=0.10, ax_rect=None):
    """
    Draw a horizontal legend panel at the bottom of a figure.

    Layout:  All 4 rows (header, Fed, Starved, footer) span the full figure width.
    8 symbol columns evenly distributed, footer centered below.

    Args:
        fig: Matplotlib Figure
        styles: Style dictionary (from styles.yaml / get_style())
        line_y: y-position for the separator line in figure coords
        ax_rect: [left, bottom, width, height] for the legend axes
                 (default [0.03, 0.02, 0.95, 0.095])
    """

    if ax_rect is None:
        ax_rect = [0.03, 0.02, 0.95, 0.11]

    # ── Separator line ──
    fig.add_artist(plt.Line2D([0.05, 0.97], [line_y, line_y],
                              transform=fig.transFigure, color='0.75',
                              linewidth=0.8, alpha=0.5, zorder=0))

    # Legend axes: full width, sits well below separator
    ax = fig.add_axes(ax_rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # ── Column definitions ──
    columns = [
        ('T app',   'fed_taste_positive',       'starved_taste_positive'),
        ('T avr',   'fed_taste_negative',        'starved_taste_negative'),
        ('O app',   'fed_odor_positive',         'starved_odor_positive'),
        ('O avr',   'fed_odor_negative',         'starved_odor_negative'),
        ('OT app',  'fed_odor_pos_taste_pos',    'starved_odor_pos_taste_pos'),
        ('T$^{+}$O$^{-}$', 'fed_odor_neg_taste_pos', 'starved_odor_neg_taste_pos'),
        ('T$^{-}$O$^{+}$', 'fed_odor_pos_taste_neg', 'starved_odor_pos_taste_neg'),
        ('OT avr',  'fed_odor_neg_taste_neg',    'starved_odor_neg_taste_neg'),
    ]

    ncols = len(columns)
    # Full-width span for all 8 symbol columns
    x_start, x_end = 0.08, 0.92
    xs = np.linspace(x_start, x_end, ncols)

    # y positions — packed tight
    y_header = 0.95
    y_fed    = 0.65
    y_stv    = 0.35
    y_footer = 0.05
    ms = 14

    # Row labels (left of first column)
    ax.text(x_start - 0.045, y_fed, 'Fed', ha='right', va='center', fontsize=FONT_SIZES['legend_panel'], weight='bold')
    ax.text(x_start - 0.045, y_stv, 'Stv', ha='right', va='center', fontsize=FONT_SIZES['legend_panel'], weight='bold')

    for i, (header, fed_key, stv_key) in enumerate(columns):
        x = xs[i]

        # Header — bold, dark
        ax.text(x, y_header, header, ha='center', va='center',
                fontsize=FONT_SIZES['legend_panel'], weight='bold', color='0.2')

        fed_s = styles[fed_key]
        stv_s = styles[stv_key]

        # ── Draw Fed symbol ──
        if fed_s.get('bicolor', False):
            ax.plot(x, y_fed, marker=HALF_CIRCLE_LEFT, ms=ms,
                    markerfacecolor=fed_s['left_color'], markeredgecolor=fed_s['left_edgecolor'],
                    markeredgewidth=1.2, clip_on=False, zorder=5)
            ax.plot(x, y_fed, marker=HALF_CIRCLE_RIGHT, ms=ms,
                    markerfacecolor=fed_s['right_color'], markeredgecolor=fed_s['right_edgecolor'],
                    markeredgewidth=1.2, clip_on=False, zorder=5)
        else:
            ax.plot(x, y_fed, 'o', ms=ms,
                    markerfacecolor=fed_s['color'], markeredgecolor=fed_s['edgecolor'],
                    markeredgewidth=1.2, clip_on=False, zorder=5)

        # ── Draw Starved symbol — thicker edges ──
        stv_ew = 2.5
        if stv_s.get('bicolor', False):
            ax.plot(x, y_stv, marker=HALF_CIRCLE_LEFT, ms=ms,
                    markerfacecolor=stv_s['left_color'], markeredgecolor=stv_s['left_edgecolor'],
                    markeredgewidth=stv_ew, clip_on=False, zorder=5)
            ax.plot(x, y_stv, marker=HALF_CIRCLE_RIGHT, ms=ms,
                    markerfacecolor=stv_s['right_color'], markeredgecolor=stv_s['right_edgecolor'],
                    markeredgewidth=stv_ew, clip_on=False, zorder=5)
        else:
            ax.plot(x, y_stv, 'o', ms=ms,
                    markerfacecolor=stv_s['color'], markeredgecolor=stv_s['edgecolor'],
                    markeredgewidth=stv_ew, clip_on=False, zorder=5)

    # ── Footer row: encoding rules + modality swatches, evenly spaced across full width ──
    footer_parts = [
        ('text',   dict(s=u'\u25cf  Filled = Fed',       color='0.4')),
        ('text',   dict(s=u'\u25cb  Open = Starved',     color='0.4')),
        ('text',   dict(s=u'\u25d1  Split = Conflict',   color='0.4')),
        ('swatch', dict(fc=styles['taste']['color'],      label='Taste')),
        ('swatch', dict(fc=styles['odor']['color'],       label='Odor')),
        ('swatch', dict(fc=styles['odor_taste']['color'], label='O+T')),
    ]
    n_footer = len(footer_parts)
    fxs = np.linspace(x_start, x_end, n_footer)

    for fx, (ftype, fkw) in zip(fxs, footer_parts):
        if ftype == 'text':
            ax.text(fx, y_footer, fkw['s'], ha='center', va='center',
                    fontsize=FONT_SIZES['legend_panel'], color=fkw['color'])
        elif ftype == 'swatch':
            ax.plot(fx - 0.015, y_footer, 's', ms=10, markerfacecolor=fkw['fc'],
                    markeredgecolor=fkw['fc'], clip_on=False)
            ax.text(fx + 0.01, y_footer, fkw['label'], ha='left', va='center',
                    fontsize=FONT_SIZES['legend_panel'], color='0.3')