import torch
import logging
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from typing import Optional
from src.models.model_io import save_model



def train_seq_seq_Classifier(model: torch.nn.Module,
                             device: torch.device,
                             train_loader: torch.utils.data.DataLoader,
                             val_loader: torch.utils.data.DataLoader,
                             num_epochs: int,
                             lr: float,
                             weight_decay: float,
                             start_epoch: int = 0,
                             criterion: torch.nn.Module = CrossEntropyLoss(),
                             logger: Optional[logging.Logger] = None, 
                             save_best_from: Optional[int] = None,
                             best_output_path: Optional[str] = None,
                             model_class: Optional[type] = None,
                             params: Optional[dict] = None,
                             monitor: str = "val_acc",   # or "val_loss"
                             mode: str = "max"):
    """
    Trains a sequence-to-sequence classifier model.
    
    Returns:
        model: Trained model
        train_loss: List of training losses per epoch
        val_loss: List of validation losses per epoch
        train_acc: List of training accuracies per epoch
        val_acc: List of validation accuracies per epoch
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    # --- INIT BEST TRACKING (NEW) ---
    track_best = (save_best_from is not None and best_output_path is not None
                  and model_class is not None and params is not None)
    have_best = False
    mode = mode.lower()
    if monitor not in ("val_acc", "val_loss"):
        monitor = "val_acc"
    if mode not in ("max", "min"):
        mode = "max" if monitor == "val_acc" else "min"
    # best_metric exists from the start
    best_metric = float("-inf") if mode == "max" else float("inf")
    save_after = int(save_best_from or 0)

    current_epoch = start_epoch  # for safe error logging


    try:
        for epoch in range(start_epoch, num_epochs):
            current_epoch = epoch + 1
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            for images, labels in train_loader:
                labels = labels.to(device)
                images = images.float().to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)

            # Validation
            model.eval()
            v_loss, v_correct, v_total = 0.0, 0, 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    labels = labels.to(device)
                    images = images.float().to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    v_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    v_total += labels.size(0)
                    v_correct += (predicted == labels).sum().item()

            val_epoch_loss = v_loss / len(val_loader)
            val_epoch_acc = 100 * v_correct / v_total
            val_loss.append(val_epoch_loss)
            val_acc.append(val_epoch_acc)

            logger.info(
                f"Epoch {epoch + 1}, "
                f"Train Loss: {train_loss[-1]:.4f}, "
                f"Train Acc: {train_acc[-1]:.2f}%, "
                f"Val Loss: {val_loss[-1]:.4f}, "
                f"Val Acc: {val_acc[-1]:.2f}%"
            )

            # --- save-best-after-warmup (minimal & safe)
            # logger.info(f"track_best: {track_best}, current_epoch: {current_epoch}, save_after: {save_after}")
            if track_best and current_epoch >= save_after:      # <-- gate before comparing
                metric = val_epoch_acc if monitor == "val_acc" else val_epoch_loss
                improved = (metric > best_metric) if mode == "max" else (metric < best_metric)
                # logger.info(f"improved: {improved}, metric: {metric}, best_metric: {best_metric}")
                if improved:
                    best_metric = metric
                    # logger.info(f"[best] epoch {current_epoch} improved {monitor} → {metric:.6f}; saving.")
                    have_best = True
                    save_model(model=model,
                               epoch=current_epoch,
                               model_class=model_class,
                               params=params,
                               output_path=best_output_path,   # e.g. ".../models/best"
                               train_loss=train_loss,
                               val_loss=val_loss,
                               train_acc=train_acc,
                               val_acc=val_acc,
                               logger=logger)
                    logger.info(f"[best] epoch {current_epoch} improved {monitor} → {metric:.6f}; saved.")
        if track_best and current_epoch >= save_after and not have_best:
            logger.info(f"[best] No improvement after epoch {save_after}.")

    except Exception as e:
        logger.error(f"Training failed at epoch {epoch + 1}: {str(e)}", exc_info=True)
        # Return the model and losses up to the point of failure
        return model, train_loss, val_loss, train_acc, val_acc

    return model, train_loss, val_loss, train_acc, val_acc
