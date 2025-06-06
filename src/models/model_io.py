import os
import torch
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger("ModelIO")


def save_model(
        model: torch.nn.Module,
        epoch: int = 0,
        model_class: Optional[type] = None,
        params: Optional[dict] = None,
        output_path: str = '',
        train_loss: Optional[List[float]] = None,
        val_loss: Optional[List[float]] = None,
) -> None:
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_params = {k: v for k, v in (params or {}).items() if k != 'num_epochs'}
    param_str = "_".join([f"{key}={value}" for key, value in model_params.items()])
    model_save_path = os.path.join(output_path, f"{model_class.__name__}_{param_str}.pth")

    # Save the model state and metadata
    save_dict = {
        'model_state_dict': model.state_dict(),  # Model weights
        'epoch': epoch,  # Current epoch
        'model_class': model_class.__name__,  # Model class name
        'params': params,  # Model parameters for reconstruction
        'train_loss': train_loss,  # Training loss history
        'val_loss': val_loss,  # Validation loss history
    }

    try:
        torch.save(save_dict, model_save_path)
        logging.info(f"Model saved to {model_save_path}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")

def load_model(
    model_class: type,
    params: dict,
    output_path: str,
    device: torch.device,
) -> Tuple[Optional[torch.nn.Module], int, List[float], List[float]]:
    model_params = {k: v for k, v in (params or {}).items() if k != 'num_epochs'}
    param_str = "_".join([f"{key}={value}" for key, value in model_params.items()])
    model_save_path = os.path.join(output_path, f"{model_class.__name__}_{param_str}.pth")
    logging.info(f"Loading model from {model_save_path}")

    if os.path.exists(model_save_path):
        try:
            checkpoint = torch.load(model_save_path, map_location=device)
            model_params = {k: v for k, v in (model_params or {}).items() if
                            k not in ['lr', 'weight_decay', 'criterion']}
            # Initialize the model
            model = model_class(**model_params).to(device)

            # Dynamically adjust keys in state_dict
            saved_state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for key in saved_state_dict.keys():
                # Map old keys to new keys (example: adjust naming convention)
                new_key = key.replace('transformer_encoder.layers', 'transformer_encoder')  # Update as needed
                new_state_dict[new_key] = saved_state_dict[key]

            # Load the state_dict into the model
            model.load_state_dict(new_state_dict, strict=False)

            # Extract other metadata
            start_epoch = checkpoint.get('epoch', 0)
            train_loss = checkpoint.get('train_loss', [])
            val_loss = checkpoint.get('val_loss', [])

            logging.info(f"Model loaded from {model_save_path}, starting from epoch {start_epoch}")
            return model, start_epoch, train_loss, val_loss
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return None, 0, [], []
    else:
        logging.info(f"Model file not found at {model_save_path}")
        return None, 0, [], []
