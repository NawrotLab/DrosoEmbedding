import mlflow
import numpy as np
import torch
import os
import pickle
from sklearn.manifold import TSNE
import yaml
# import umap

def log_params_recursive(d):
    for k, v in d.items():
        if isinstance(v, dict):
            log_params_recursive(v)
        elif isinstance(v, (int, float, str, bool)):
            mlflow.log_param(k, v)

def get_latent_space(model, data_loader, device, return_cnn_latent=False):
    model.eval()
    latent_space, latent_labels = [], []
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences = sequences.float().to(device)
            labels = labels.to(device)
            # Pass data through the model and collect latent features
            latent_features = model(sequences, return_cnn_latent=return_cnn_latent)
            latent_space.append(latent_features.cpu().numpy())
            latent_labels.append(labels.cpu().numpy())

    # Stack collected features and labels
    latent_features_np = np.concatenate(latent_space, axis=0)
    latent_labels_np = np.concatenate(latent_labels, axis=0)
    return latent_features_np, latent_labels_np

def get_predictions(model, data_loader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences = sequences.float().to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(sequences)
            predicted = torch.argmax(outputs, dim=1)
            predictions.append(predicted.cpu().numpy())
            true_labels.append(labels.cpu().numpy())
    # Stack collected predictions and labels
    predictions_np = np.concatenate(predictions, axis=0)
    true_labels_np = np.concatenate(true_labels, axis=0)
    return predictions_np, true_labels_np 

def compute_tsne(latent_features, perplexity=30, random_state=42):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    return tsne.fit_transform(latent_features)

def load_all_results(base_dir, TASK_CLASS_NAMES):
    """Load eval pickles plus attach class names.
    
    Args:
        base_dir: Base directory containing task folders
        TASK_CLASS_NAMES: Dictionary mapping task names to class names
        
    Returns:
        dict: Nested dictionary with structure:
            {
                'task_name': {
                    'run_name': { ... evaluation results ... },
                    'dim_runs': {
                        'dim_4': [run1_data, run2_data, ...],
                        'dim_8': [run1_data, run2_data, ...],
                        ...
                    },
                    '__class_names__': [...]
                },
                ...
            }
    """
    results = {}
    for task in sorted(os.listdir(base_dir)):
        task_path = os.path.join(base_dir, task)
        if not os.path.isdir(task_path):
            continue
            
        results[task] = {}
        dim_runs_path = os.path.join(task_path, 'dim_runs')
        
        # Load regular runs
        for run_name in sorted(os.listdir(task_path)):
            if run_name == 'dim_runs':
                continue  # Handle dim_runs separately
                
            pkl = os.path.join(task_path, run_name, 'evaluation_results.pkl')
            if os.path.isfile(pkl):
                with open(pkl, 'rb') as f:
                    results[task][run_name] = pickle.load(f)
        
        # Load dimension runs if they exist
        if os.path.exists(dim_runs_path):
            results[task]['dim_runs'] = {}
            for filename in sorted(os.listdir(dim_runs_path)):
                if not filename.endswith('_evalResults.pkl'):
                    continue
                    
                # Parse dimension and run number from filename
                # Format: 'SpeedSeed6_trf4_1_evalResults.pkl' -> dim=4, run=1
                try:
                    parts = filename.split('_')
                    dim = int(parts[1].replace('trf', ''))
                    run_num = int(parts[2].split('_')[0])
                    
                    # Load the results
                    pkl_path = os.path.join(dim_runs_path, filename)
                    with open(pkl_path, 'rb') as f:
                        run_data = pickle.load(f)
                        
                    # Store with dimension and run info
                    dim_key = f'dim_{dim}'
                    if dim_key not in results[task]['dim_runs']:
                        results[task]['dim_runs'][dim_key] = []
                    results[task]['dim_runs'][dim_key].append({
                        'run': run_num,
                        'data': run_data,
                        'dimension': dim
                    })
                except (IndexError, ValueError) as e:
                    print(f"Warning: Could not parse dimension/run from {filename}: {e}")
                    continue
        
        # Add class names if available
        if task in TASK_CLASS_NAMES:
            results[task]['__class_names__'] = TASK_CLASS_NAMES[task]
            
    return results

def get_style(style = "stylesD"):
    
    with open(f'src/visualization/{style}.yaml', "r") as f:
        styles = yaml.safe_load(f)["styles"]

    # Task-specific class names
    TASK_CLASS_NAMES = {
        'MetabolicState_2': ["Starved", "Fed"],
        'State_Modality_6': [ "Odor (S)", "Odor (F)", "Taste (S)", "Taste (F)", "Odor + Taste (S)", "Odor + Taste (F)"],
        'State_Modality_Valence_16': [
            "O$^{+}$ (S)", "O$^{-}$ (S)", "O$^{+}$ (F)", "O$^{-}$ (F)", 
            "T$^{+}$ (S)", "T$^{-}$ (S)", "T$^{+}$ (F)", "T$^{-}$ (F)", 
            "O$^{+}$+T$^{+}$ (S)", "O$^{-}$+T$^{-}$ (S)", "O$^{-}$+T$^{+}$ (S)", "O$^{+}$+T$^{-}$ (S)", 
            "O$^{+}$+T$^{+}$ (F)", "O$^{-}$+T$^{-}$ (F)", "O$^{-}$+T$^{+}$ (F)", "O$^{+}$+T$^{-}$ (F)"]
    }
    TASK_COLORS = {
        'MetabolicState_2': [styles['starved']['color'], styles['fed']['color']],
        'State_Modality_6': [
            styles['starved_odor']['color'], styles['fed_odor']['color'],
            styles['starved_taste']['color'], styles['fed_taste']['color'],
            styles['starved_odor_taste']['color'], styles['fed_odor_taste']['color']],

        'State_Modality_Valence_16': [
            styles['starved_odor_positive']['color'], styles['starved_odor_negative']['color'],
            styles['fed_odor_positive']['color'], styles['fed_odor_negative']['color'],

            styles['starved_taste_positive']['color'], styles['starved_taste_negative']['color'],
            styles['fed_taste_positive']['color'], styles['fed_taste_negative']['color'],

            styles['starved_odor_pos_taste_pos']['color'], styles['starved_odor_neg_taste_neg']['color'],
            styles['starved_odor_neg_taste_pos']['color'], styles['starved_odor_pos_taste_neg']['color'],

            styles['fed_odor_pos_taste_pos']['color'], styles['fed_odor_neg_taste_neg']['color'],
            styles['fed_odor_neg_taste_pos']['color'], styles['fed_odor_pos_taste_neg']['color']]
    }

    TASK_EDGECOLORS = {
        'MetabolicState_2': [styles['starved']['edgecolor'], styles['fed']['edgecolor']],
        'State_Modality_6': [
            styles['starved_odor']['edgecolor'], styles['fed_odor']['edgecolor'],
            styles['starved_taste']['edgecolor'], styles['fed_taste']['edgecolor'],
            styles['starved_odor_taste']['edgecolor'], styles['fed_odor_taste']['edgecolor']],

        'State_Modality_Valence_16': [
            styles['starved_odor_positive']['edgecolor'], styles['starved_odor_negative']['edgecolor'],
            styles['fed_odor_positive']['edgecolor'], styles['fed_odor_negative']['edgecolor'],

            styles['starved_taste_positive']['edgecolor'], styles['starved_taste_negative']['edgecolor'],
            styles['fed_taste_positive']['edgecolor'], styles['fed_taste_negative']['edgecolor'],

            styles['starved_odor_pos_taste_pos']['edgecolor'], styles['starved_odor_neg_taste_neg']['edgecolor'],
            styles['starved_odor_neg_taste_pos']['edgecolor'], styles['starved_odor_pos_taste_neg']['edgecolor'],

            styles['fed_odor_pos_taste_pos']['edgecolor'], styles['fed_odor_neg_taste_neg']['edgecolor'],
            styles['fed_odor_neg_taste_pos']['edgecolor'], styles['fed_odor_pos_taste_neg']['edgecolor']
        ]
    }

    TASK_SHAPES = {
        'MetabolicState_2': [styles['starved']['shape'], styles['fed']['shape']],
        'State_Modality_6': [
            styles['starved_odor']['shape'], styles['starved_taste']['shape'],
            styles['starved_odor_taste']['shape'], styles['fed_odor']['shape'],
            styles['fed_taste']['shape'], styles['fed_odor_taste']['shape']
        ],
        'State_Modality_Valence_16': [
            styles['starved_odor_positive']['shape'], styles['starved_odor_negative']['shape'],
            styles['fed_odor_positive']['shape'], styles['fed_odor_negative']['shape'],

            styles['starved_taste_positive']['shape'], styles['starved_taste_negative']['shape'],
            styles['fed_taste_positive']['shape'], styles['fed_taste_negative']['shape'],

            styles['starved_odor_pos_taste_pos']['shape'], styles['starved_odor_neg_taste_neg']['shape'],
            styles['starved_odor_neg_taste_pos']['shape'], styles['starved_odor_pos_taste_neg']['shape'],
            styles['fed_odor_pos_taste_pos']['shape'], styles['fed_odor_neg_taste_neg']['shape'],
            styles['fed_odor_neg_taste_pos']['shape'], styles['fed_odor_pos_taste_neg']['shape']
        ]
    }

    return styles, TASK_CLASS_NAMES, TASK_COLORS, TASK_EDGECOLORS, TASK_SHAPES
