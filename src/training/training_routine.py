import torch
from torch.nn import CrossEntropyLoss
import logging
import torch.optim as optim
from src.utils.config import training_config
from pathlib import Path
from typing import Optional, Dict, Any

# from src.analysis.learning_dynamics import EnhancedDynamicsAnalyzer
from src.utils import config 

logger = logging.getLogger(training_config["run_name"])


def train_seq_seq_Classifier(model: torch.nn.Module,
                             # optimizer: torch.optim.Optimizer,
                             device:torch.device,
                             train_loader: torch.utils.data.DataLoader,
                             val_loader: torch.utils.data.DataLoader,
                             num_epochs: int,
                             lr:float,
                             weight_decay:float,
                             # save_dir: str,
                             start_epoch: int = 0,
                             criterion: torch.nn.Module = CrossEntropyLoss()):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss = []
    val_loss = []
    #set_seed(42)
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            labels = labels.to(device)
            images = images.float().to(device)

            optimizer.zero_grad()  # Reset the gradients to zero
            outputs = model(images)
            #predicted = torch.argmax(outputs, dim = 1)
            loss = criterion(outputs, labels)

            loss.backward()  # Preform backpropagation
            optimizer.step()  # optimize loss parameters
            running_loss += loss.item()


        # Track training loss
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

            # Track validation loss
            val_loss.append(v_loss / len(val_loader))
        logger.info(f"Epoch {epoch + 1}, Training Loss: {train_loss[-1]}, Validation Loss: {val_loss[-1]}")

    return model, train_loss, val_loss