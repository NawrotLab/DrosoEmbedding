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
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    model.to(device)
    torch.compile(model) # VR: torch compile is supposed to speed up the training
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss = []
    val_loss = []

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                labels = labels.to(device)
                images = images.float().to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss.append(running_loss / len(train_loader))

            # Validation
            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    labels = labels.to(device)
                    images = images.float().to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    v_loss += loss.item()

            val_loss.append(v_loss / len(val_loader))

            logger.info(f"Epoch {epoch + 1}, Training Loss: {train_loss[-1]:.4f}, Validation Loss: {val_loss[-1]:.4f}")

    except Exception as e:
        logger.error(f"Training failed at epoch {epoch + 1}: {str(e)}", exc_info=True)
        # Return the model and losses up to the point of failure
        return model, train_loss, val_loss

    return model, train_loss, val_loss
