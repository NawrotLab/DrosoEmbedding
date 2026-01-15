from src.models.model_io import load_model
from src.utils.config_loader import load_config
from torch.utils.data import DataLoader
from src.data.dataset import CustomDataset
import pickle
import os
import numpy as np
import torch
from src.utils.logger import setup_logger
from src.models.cnn_transformer import CNN_Transformer

config = load_config()
logger = setup_logger(task_name=config['run_id'], log_dir="logs/neuropils_importance_cofficients")

neuropils = ['AL', 'MB', 'PENP', 'VLNP', 'CX', 'GNG', 'LX', 'SNP', 'INP', 'LH', 'OL', 'VMNP']

run_id = config['run_id']
config_path = f"{config['paths']['results_root']}/config.pkl"
paths = config['paths']
model_params = config['model']['parameters']
training_params = config['training']

# Load config from pickle file
with open(config_path, 'rb') as f:
    config = pickle.load(f)


# Load test data
with open(paths["pickle_path"], 'rb') as f:
    _, _, X_test, _, _, Y_test = pickle.load(f)
    

test_dataset = CustomDataset( X_test, Y_test, transform=True,seq_length=model_params['seq_len'], seq_steps=model_params['seq_steps'], allTs_path=config['paths']['allTs_path'])
test_loader = DataLoader(test_dataset, batch_size=training_params['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

logger.info(f"Test loader: {len(test_loader)}")
logger.info(f"Test example path: {X_test[0]}")

logger.info(f"Locating Model at {paths['models']}")
logger.info(f"Loading Model with parameters: {model_params} in {config['device']}.")

classifier, _, _, _, _, _ = load_model(
    CNN_Transformer, model_params, paths['models'], config['device'], logger
)

if classifier is None:
    raise FileNotFoundError("Trained model not found.")

# 1) Create output directory
outdir = "results/neuropil_importance_coefficients"
os.makedirs(outdir, exist_ok=True)
logger.info(f"Output directory created: {outdir}")

# 2) Run inference on test_loader to collect raw logits
classifier.eval()
all_logits = []
all_labels = []

logger.info("Starting inference on test_loader...")
with torch.no_grad():
    for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
        X_batch = X_batch.to(config['device'])
        output = classifier(X_batch)
        
        # Handle tuple output (model may return (logits, ...) or just logits)
        if isinstance(output, tuple):
            logits_batch = output[0]
            logger.info(f"Batch {batch_idx}: Model returned tuple, using first element. Shape: {logits_batch.shape}")
        else:
            logits_batch = output
            if batch_idx == 0:
                logger.info(f"Batch {batch_idx}: Model returned tensor. Shape: {logits_batch.shape}")
        
        all_logits.append(logits_batch.cpu().numpy())
        all_labels.append(y_batch.numpy())

# Concatenate all batches
logits = np.concatenate(all_logits, axis=0)
y_true = np.concatenate(all_labels, axis=0)

# 3) Save intermediate outputs
np.save(f"{outdir}/tmp_logits.npy", logits)
np.save(f"{outdir}/tmp_ytrue.npy", y_true)
logger.info(f"Saved logits to {outdir}/tmp_logits.npy")
logger.info(f"Saved y_true to {outdir}/tmp_ytrue.npy")

# 4) Log details
logger.info(f"Number of batches processed: {len(all_logits)}")
logger.info(f"Logits shape: {logits.shape}")
logger.info(f"First 5 logits rows:\n{logits[:5]}")
unique, counts = np.unique(y_true, return_counts=True)
logger.info(f"Unique y_true values and counts: {dict(zip(unique, counts))}")