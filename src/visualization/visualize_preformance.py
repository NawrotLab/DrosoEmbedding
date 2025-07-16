import matplotlib.pyplot as plt
import torch
import numpy as np 
from sklearn.manifold import TSNE
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
import seaborn as sns
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
import textwrap

import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm

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
                        batch_size, learning_rate, output_path, ylim=None):
    """
    Plot and save the training and validation loss curves.

    Args:
    - training_loss: List of training loss values per epoch.
    - validation_loss: List of validation loss values per epoch.
    - dataID: Identifier for the dataset.
    - model_name: Name of the model used.
    - batch_size: Batch size used during training.
    - learning_rate: Learning rate used during training.
    - output_path: Directory where the plot will be saved.
    - ylim: Optional tuple for y-axis limits.
    """
    
    num_epochs = len(training_loss)
    title = f"{model_name}: {dataID}"
    subtitle = f"BS: {batch_size}; LR: {learning_rate}; Epochs: {num_epochs}"

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


# def plot_confusion_matrix(cl_name, cm, class_names, output_path, dataID, hyperparameters=None):
#     """
#     Plot and save a confusion matrix with overall accuracy in the title.

#     Args:
#     - cl_name: Classifier name or title.
#     - cm: Confusion matrix array.
#     - class_names: List of class names for the matrix.
#     - output_path: Directory where the plot will be saved.
#     - dataID: Identifier for the dataset.
#     - hyperparameters: Optional dictionary of hyperparameters for the title.
#     """
#     if len(class_names) < 4:
#         fontsize_annot = 30
#         fontsize_ticks = 30
#         fontsize_axis = 30
#         fontsize_title = 30
#     else:
#         fontsize_annot = 15
#         fontsize_ticks = 18
#         fontsize_axis = 18
#         fontsize_title = 26


#     plt.figure(figsize=(12, 12))

#     # Calculate overall accuracy
#     total_correct = np.trace(cm)  # Sum of diagonal elements
#     total_samples = np.sum(cm)
#     accuracy = (total_correct / total_samples) * 100

#     # Build title with overall accuracy
#     hyperparams_str = ', '.join([f"{key}={value}" for key, value in hyperparameters.items()]) if hyperparameters else ''
#     title = f'Test Set Accuracy: {accuracy:.2f}%'
#     # title = f'{cl_name} (Accuracy: {accuracy:.2f}%): {hyperparams_str}'
#     wrapped_title = "\n".join(textwrap.wrap(title, width=70))
#     plt.title(wrapped_title, fontsize=fontsize_title, pad=20)

#     # Normalize confusion matrix to percentages for better visualization
#     cm_percentage = (cm / cm.sum(axis=1, keepdims=True)) * 100
#     annot_labels = np.array([[f"{percent:.1f}%\n({count})" for percent, count in zip(row_percent, row)]
#                              for row_percent, row in zip(cm_percentage, cm)])





#     # cadetblue_cmap = LinearSegmentedColormap.from_list("CadetBlue", ["#d1e8e2", "#5f9ea0", "#2a5050"])
#     sns.heatmap(cm_percentage,
#                 # annot=annot_labels,
#                 fmt="",
#                 cmap='Blues',
#                 xticklabels=class_names,
#                 yticklabels=class_names,
#                 cbar=False,
#                 vmin=0, vmax=100,
#                 annot_kws={"size": fontsize_annot})  # Increase annotation font size

#     # Increase font size of x and y axis labels
#     plt.xticks(fontsize=fontsize_ticks)  # Adjust tick labels font size
#     plt.yticks(fontsize=fontsize_ticks)
#     plt.xticks(rotation=0)  # Rotate X labels
#     plt.yticks(rotation=90)  # Rotate X labels

#     plt.xlabel('Predicted Label', fontsize=fontsize_axis, labelpad=20)
#     plt.ylabel('True Label', fontsize=fontsize_axis, labelpad=20)

#     # Adjust layout for long class names
#     plt.tight_layout()  # Automatically adjust layout
#     plt.subplots_adjust(left=0.2, bottom=0.2)  # Leave extra space for labels if needed

#     # Save the confusion matrix plot
#     out_path = os.path.join(output_path, f"{len(class_names)}Cls_ConfusionMatrix.png")

#     # out_path = os.path.join(output_path, f"{dataID}_{cl_name}_ConfusionMatrix.png")
#     #plt.savefig(out_path)
#     plt.savefig(out_path, bbox_inches='tight', pad_inches=0.3, dpi=300)
#     plt.close()
#     print(f"Confusion matrix saved to {out_path}")

# def plot_confusion_matrix(cl_name, cm, class_names, output_path, dataID, hyperparameters=None):
#     """
#     Plot and save a confusion matrix with overall accuracy in the title.

#     Args:
#     - cl_name: Classifier name or title.
#     - cm: Confusion matrix array.
#     - class_names: List of class names for the matrix.
#     - output_path: Directory where the plot will be saved. If None, the function returns the axis instead of saving.
#     - dataID: Identifier for the dataset.
#     - hyperparameters: Optional dictionary of hyperparameters for the title.
#     """
#     # Determine figure saving behavior
#     save_fig = output_path is not None

#     # Set font sizes based on number of classes
#     if len(class_names) < 4:
#         fontsize_annot = 30
#         fontsize_ticks = 30
#         fontsize_axis = 30
#         fontsize_title = 30
#     else:
#         fontsize_annot = 15
#         fontsize_ticks = 18
#         fontsize_axis = 18
#         fontsize_title = 26

#     # Create figure and axis
#     fig, ax = plt.subplots(figsize=(12, 12))

#     # Calculate overall accuracy
#     total_correct = np.trace(cm)  # Sum of diagonal elements
#     total_samples = np.sum(cm)
#     accuracy = (total_correct / total_samples) * 100

#     # Build title with overall accuracy
#     hyperparams_str = ', '.join([f"{key}={value}" for key, value in hyperparameters.items()]) if hyperparameters else ''
#     title = f'Test Set Accuracy: {accuracy:.2f}%'
#     wrapped_title = "\n".join(textwrap.wrap(title, width=70))
#     ax.set_title(wrapped_title, fontsize=fontsize_title, pad=20)

#     # Normalize confusion matrix to percentages for better visualization
#     cm_percentage = (cm / cm.sum(axis=1, keepdims=True)) * 100
#     print(cm_percentage)

#     # Annotate with counts and percentages
#     annot_labels = np.array([
#         [f"{percent:.1f}%\n({count})" for percent, count in zip(row_percent, row)]
#         for row_percent, row in zip(cm_percentage, cm)
#     ])

#     # Plot heatmap
#     sns.heatmap(
#         cm_percentage,
#         annot=annot_labels,
#         fmt="",
#         cmap='Blues',
#         xticklabels=class_names,
#         yticklabels=class_names,
#         cbar=False,
#         vmin=0, vmax=100,
#         annot_kws={"size": fontsize_annot},
#         ax=ax
#     )

#     # Format axis labels and ticks
#     ax.set_xlabel('Predicted Label', fontsize=fontsize_axis, labelpad=20)
#     ax.set_ylabel('True Label', fontsize=fontsize_axis, labelpad=20)
#     ax.tick_params(axis='x', labelsize=fontsize_ticks, rotation=0)
#     ax.tick_params(axis='y', labelsize=fontsize_ticks, rotation=90)

#     # Adjust layout
#     fig.tight_layout()
#     fig.subplots_adjust(left=0.2, bottom=0.2)

#     if save_fig:
#         # Ensure the output directory exists
#         os.makedirs(output_path, exist_ok=True)
#         # Build output file path
#         out_path = os.path.join(output_path, f"{len(class_names)}Cls_ConfusionMatrix.png")
#         # Save the confusion matrix plot
#         fig.savefig(out_path, bbox_inches='tight', pad_inches=0.3, dpi=300)
#         plt.close(fig)
#         print(f"Confusion matrix saved to {out_path}")
#     else:
#         return ax

def plot_confusion_matrix(cl_name,
                          cm,
                          class_names,
                          output_path,
                          dataID,
                          hyperparameters=None,
                          ax=None,
                          annot=True,
                          cbar=False):
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

    # Draw heatmap
    sns.heatmap(
        cm_pct,
        annot=annot_data,
        fmt="",
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names if standalone else [],
        cbar=cbar,
        vmin=0,
        vmax=100,
        annot_kws=annot_kws,
        ax=ax
    )

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
      ax: Optional Axes to draw on (subplot mode)
      title: Optional subplot title
      legend: whether to show legend
      draw_axis: whether to draw x/y labels
      draw_title: whether to draw the title
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 10))

    palette = sns.color_palette("tab20", len(class_names))
    for cls in np.unique(labels_np):
        idx = labels_np == cls
        ax.scatter(latent_2d[idx, 0], latent_2d[idx, 1],
                   label=f"{class_names[int(cls)]} ({idx.sum()})",
                   alpha=0.6, s=6,
                   color=palette[int(cls)], edgecolors='none')

    # determine limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', xmin))
    ax.spines['bottom'].set_position(('data', ymin))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

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

# def plot_tsne(model, data_loader, device, cl_name, output_path, dataID, class_names, ax=None, hyperparameters=None):
#     """
#     Plot and save a TSNE visualization of the latent space.

#     Args:
#     - model: Trained model to extract latent features.
#     - data_loader: DataLoader containing data to visualize.
#     - device: Torch device (e.g., 'cpu' or 'cuda').
#     - cl_name: Classifier name or title.
#     - output_path: Directory where the plot will be saved.
#     - dataID: Identifier for the dataset.
#     - class_names: List of class names.
#     - hyperparameters: Optional dictionary of hyperparameters for the title.
#     """
#     model.eval()
#     latent_features_list, labels_list = [], []

#     # Extract latent features and labelsno i dont understand
#     with torch.no_grad():
#         for sequences, labels in data_loader:
#             sequences = sequences.float().to(device)
#             labels = labels.to(device)

#             # Pass data through the model and collect latent features
#             latent_features = model(sequences, return_latent=True)
#             latent_features_list.append(latent_features.cpu().numpy())
#             labels_list.append(labels.cpu().numpy())

#     # Stack collected features and labels
#     latent_features_np = np.concatenate(latent_features_list, axis=0)
#     labels_np = np.concatenate(labels_list, axis=0)



#     # Apply TSNE
#     tsne = TSNE(n_components=2, random_state=42, perplexity=30)
#     latent_2d = tsne.fit_transform(latent_features_np)

#     num_classes = len(class_names)
#     colors = sns.color_palette("tab20", num_classes)  

#     if ax is None:
#         fig, ax = plt.subplots(figsize=(12, 10))
#         standalone = True
#     else:
#         standalone = False

#     unique_labels = np.unique(labels_np)

#     for i, class_label in enumerate(unique_labels):
#         indices = labels_np == class_label
#         count = np.sum(indices)
#         ax.scatter(latent_2d[indices, 0], latent_2d[indices, 1],
#                    label=f"{class_names[i]} ({count})", alpha=0.5, s=50,
#                    color=colors[i], edgecolors='gray', linewidth=0.5)

#     ax.set_xlim([-110,110])
#     ax.set_ylim([-110, 110])
#     ax.tick_params(labelsize=22)
#     ax.set_xlabel('TSNE Component 1', fontsize=26)
#     ax.set_ylabel('TSNE Component 2', fontsize=26)
#     #plt.legend(loc='upper right', fontsize=16, markerscale=2, scatterpoints=1)

#     if standalone and output_path:
#         out_path = os.path.join(output_path, f"{len(class_names)}Cls_LatentSpace_TSNE.png")
#         fig.savefig(out_path)
#         plt.close(fig)
#         print(f"TSNE plot saved to {out_path}")

#     if standalone:
#         return fig, ax
#     else:
#         return ax


# def plot_umap(model, data_loader, device, cl_name, output_path, dataID, class_names, hyperparameters=None):
#     import umap
#     """
#     Plot and save a UMAP visualization of the latent space.

#     Args:
#     - model: Trained model to extract latent features.
#     - data_loader: DataLoader containing data to visualize.
#     - device: Torch device (e.g., 'cpu' or 'cuda').
#     - cl_name: Classifier name or title.
#     - output_path: Directory where the plot will be saved.
#     - dataID: Identifier for the dataset.
#     - class_names: List of class names.
#     - hyperparameters: Optional dictionary of hyperparameters for the title.
#     """
#     model.eval()
#     latent_features_list, labels_list = [], []

#     with torch.no_grad():
#         for sequences, labels in data_loader:
#             sequences = sequences.float().to(device)
#             labels = labels.to(device)

#             latent_features = model(sequences, return_latent=True)
#             latent_features_list.append(latent_features.cpu().numpy())
#             labels_list.append(labels.cpu().numpy())

#     latent_features_np = np.concatenate(latent_features_list, axis=0)
#     labels_np = np.concatenate(labels_list, axis=0)

#     # Apply UMAP
#     reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.1, n_neighbors=15)
#     latent_2d = reducer.fit_transform(latent_features_np)

#     colors = sns.color_palette("tab20")  
#     # if num_classes <= 10:
#     #     colors = sns.color_palette("tab10", num_classes)  # Up to 10 colors
#     # elif num_classes <= 12:
#     #     colors = sns.color_palette("Paired", num_classes)  # Exactly 12 colors
#     # elif num_classes <= 16:
#     #     colors = sns.color_palette("tab20", num_classes)  

#     plt.figure(figsize=(12, 10))
#     unique_labels = np.unique(labels_np)

#     for i, class_label in enumerate(unique_labels):
#         indices = labels_np == class_label
#         count = np.sum(indices)

#         plt.scatter(latent_2d[indices, 0], latent_2d[indices, 1],
#                     label=f"{class_names[i]} ({count})", alpha=0.5, s=50,
#                     color=colors[i], edgecolors='gray', linewidth=0.5)

#     plt.xticks(fontsize=22)
#     plt.yticks(fontsize=22)
#     plt.xlabel('UMAP Component 1', fontsize=26)
#     plt.ylabel('UMAP Component 2', fontsize=26)
#     plt.legend(loc='upper right', fontsize=16, markerscale=2, scatterpoints=1)

#     out_path = os.path.join(output_path, f"{len(class_names)}Cls_LatentSpace_UMAP.png")
#     plt.savefig(out_path)
#     plt.close()



def reshape_transform4Cam(tensor):
    # Handle tensors with 5 dimensions (e.g., from 3D models)
    # print(f"Original tensor shape: {tensor.shape}")  # Debugging

    if tensor.dim() == 5:  # Example shape: [B, C, D, H, W]
        # Collapse depth dimension (D), returning to [B, C, H, W]
        tensor = tensor.mean(dim=2)  # Take mean across the depth dimension
    # print(f"Transformed tensor shape: {tensor.shape}")  # Debugging

    # Ensure tensor is 4D: [N, C, H, W]
    return tensor



def plot_classes_cam(model, device, dataloader, class_names, output_path, cam_method = GradCAM, classifier_target_layer = 3, reshape = True):


    # 1. Convert dataloader into dictionary with respective classes
    data_by_class = dataloader2dictionary(dataloader, class_names)

    # 2. Plot Input Tensors.
    plot_InputTensors(data_by_class, output_path, 'Mean_InputTensors')
    print(f'Mean Input Tensors saved: {output_path}')

    # 3. Set Model up for Evaluation and initiate cam
    model.eval()
    target_layer =  model.cnn[classifier_target_layer]
    if reshape:
        cam = cam_method(model=model, target_layers=[target_layer], reshape_transform=reshape_transform4Cam)
    else:
        cam = cam_method(model=model, target_layers=[target_layer])

    # 4. Get all heatmaps and sort in a dictionary
    heatmaps = {class_name: [] for class_name in class_names}
    for class_idx, (class_label, class_tensor) in enumerate(data_by_class.items()):
        for image in class_tensor: #Shape: 5,1,128,128
            if len(image.shape) == 3:
                image = image.unsqueeze(0).float().to(device)  # Shape: 1,1,128,128
                # image = image.unsqueeze(0).unsqueeze(0).float().to(device) #Shape: 1,1,1,128,128
            else:
                image = image[0:1].float().to(device)  # Shape: 1,1,128,128

                # image = image[0, :, :, :].unsqueeze(0).unsqueeze(0).float().to(device) #Shape: 1,1,1,128,128
            heatmap = cam(input_tensor=image, targets=[ClassifierOutputTarget(class_idx)]) # class_label
            heatmap_2D = heatmap[0]
            heatmaps[class_label].append(heatmap_2D) #result_img

    # 5. Calculate mean Grad-CAM heatmap for each class
    # Median and Max Cams dont work as well.
    mean_class_heatmaps = {}
    for class_label, heatmap_list in heatmaps.items():
        mean_class_heatmaps[class_label] = np.mean(heatmap_list, axis =0)

    # 6. plot the mean heatmaps
    plot_MeanCam(mean_class_heatmaps, output_path, f'Mean_{cam_method.__name__}_L{classifier_target_layer}')

    return mean_class_heatmaps


def plot_MeanCam(mean_cam_dictionary, output_path, output_name = 'Mean Cam', plot_title = 'Mean Cam', figsize=(10,8)):
    """
    Plots a grid of mean arrays from a dictionary containing arrays of shape (128, 128).
    """

    num_classes = len(mean_cam_dictionary)
    cols = 4  # Fixed columns
    rows = (num_classes + cols - 1) // cols  # Compute required rows

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for idx, (title, array) in enumerate(mean_cam_dictionary.items()):
        if idx < len(axes):  # Avoid IndexError if there are more arrays than axes
            ax = axes[idx]
            ax.imshow(array, cmap='viridis')
            ax.set_title(title)
            ax.axis("off")

    # Turn off any remaining empty subplots
    for idx in range(len(mean_cam_dictionary), len(axes)):
        axes[idx].axis("off")


    plt.tight_layout()
    # plt.suptitle(plot_title, fontsize=18)
    # plt.colorbar(im, ax=axes, location='right', shrink=0.7, aspect=20)  # Add a shared colorbar

    out_name = f'{output_path}{output_name}.png'
    plt.savefig(out_name)



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


