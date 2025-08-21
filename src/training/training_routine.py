import torch
import logging
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from typing import Optional


def train_seq_seq_Classifier(model: torch.nn.Module,
                             device: torch.device,
                             train_loader: torch.utils.data.DataLoader,
                             val_loader: torch.utils.data.DataLoader,
                             num_epochs: int,
                             lr: float,
                             weight_decay: float,
                             start_epoch: int = 0,
                             criterion: torch.nn.Module = CrossEntropyLoss(),
                             logger: Optional[logging.Logger] = None):
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
    torch.compile(model) 
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

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
            v_loss = 0.0
            v_correct = 0
            v_total = 0
            
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

    except Exception as e:
        logger.error(f"Training failed at epoch {epoch + 1}: {str(e)}", exc_info=True)
        # Return the model and losses up to the point of failure
        return model, train_loss, val_loss, train_acc, val_acc

    return model, train_loss, val_loss, train_acc, val_acc
