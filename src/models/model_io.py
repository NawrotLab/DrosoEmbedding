import os
import torch
import logging
import hashlib
from typing import Optional, List, Tuple


def save_model(
    model: torch.nn.Module,
    epoch: int = 0,
    model_class: Optional[type] = None,
    params: Optional[dict] = None,
    output_path: str = '',
    train_loss: Optional[List[float]] = None,
    val_loss: Optional[List[float]] = None,
    train_acc: Optional[List[float]] = None,
    val_acc: Optional[List[float]] = None,
    logger=None,
) -> None:
    if logger is None:
        logger = logging.getLogger("ModelIO")
    os.makedirs(output_path, exist_ok=True)

    # filename (deterministic)
    model_class = model_class or model.__class__
    model_params = {k: v for k, v in (params or {}).items() if k != 'num_epochs'}
    param_str = "_".join([f"{k}={model_params[k]}" for k in sorted(model_params)])
    model_save_path = os.path.join(output_path, f"{model_class.__name__}_{param_str}.pth")

    # canonicalize state_dict (strip DDP 'module.' if present)
    sd = model.state_dict()
    if sd and all(k.startswith("module.") for k in sd.keys()):
        sd = {k[7:]: v for k, v in sd.items()}

    # checksum (exact-bytes over canonicalized sd)
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
    state_sha256 = h.hexdigest()

    save_dict = {
        'model_state_dict': sd,
        'state_sha256': state_sha256,
        'epoch': epoch,
        'model_class': model_class.__name__,
        'params': params,
        'train_loss': [] if train_loss is None else list(train_loss),
        'val_loss': [] if val_loss is None else list(val_loss),
        'train_acc': [] if train_acc is None else list(train_acc),
        'val_acc': [] if val_acc is None else list(val_acc),
    }

    try:
        torch.save(save_dict, model_save_path)
        logger.info(f"Model saved to {model_save_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")


def load_model(
    model_class: type,
    params: dict,
    output_path: str,
    device: torch.device,
    logger=None,
) -> Tuple[Optional[torch.nn.Module], int, List[float], List[float], List[float], List[float]]:
    if logger is None:
        logger = logging.getLogger("ModelIO")

    # filename (deterministic; same as save)
    model_params = {k: v for k, v in (params or {}).items() if k != 'num_epochs'}
    param_str = "_".join([f"{k}={model_params[k]}" for k in sorted(model_params)])
    model_save_path = os.path.join(output_path, f"{model_class.__name__}_{param_str}.pth")
    logger.info(f"Loading model from {model_save_path}")

    if not os.path.exists(model_save_path):
        logger.info(f"Model file not found at {model_save_path}")
        return None, 0, [], [], [], []

    try:
        ckpt = torch.load(model_save_path, map_location=device)

        # reconstruct init params (drop training-only keys if present)
        init_params = {k: v for k, v in (ckpt.get('params', params) or {}).items()
                       if k not in ['num_epochs', 'lr', 'learning_rate', 'weight_decay', 'criterion']}
        model = model_class(**init_params).to(device)

        # get sd from ckpt; also handle legacy DDP keys just in case
        sd = ckpt['model_state_dict']
        if sd and all(k.startswith("module.") for k in sd.keys()):
            sd = {k[7:]: v for k, v in sd.items()}

        # strict load (raises on mismatch)
        model.load_state_dict(sd, strict=True)

        # checksum verification
        saved_hash = ckpt.get('state_sha256')
        if saved_hash:
            h = hashlib.sha256()
            msd = model.state_dict()
            # canonicalize for hashing (strip 'module.' if ever present)
            if msd and all(k.startswith("module.") for k in msd.keys()):
                msd = {k[7:]: v for k, v in msd.items()}
            for k in sorted(msd.keys()):
                t = msd[k].detach().cpu().contiguous()
                h.update(k.encode()); h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
            loaded_hash = h.hexdigest()

            if loaded_hash == saved_hash:
                logger.info(f"Checksum verified: {loaded_hash}")
            else:
                logger.warning(f"Checksum mismatch: saved {saved_hash} vs loaded {loaded_hash}")

        train_loss = ckpt.get('train_loss', [])
        val_loss   = ckpt.get('val_loss', [])
        train_acc  = ckpt.get('train_acc', [])
        val_acc    = ckpt.get('val_acc', [])
        start_epoch = ckpt.get('epoch', len(train_loss))

        logger.info(f"Model loaded. epoch={start_epoch}, val_acc_best={max(val_acc) if val_acc else 'n/a'}")
        return model, start_epoch, train_loss, val_loss, train_acc, val_acc

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, 0, [], [], [], []