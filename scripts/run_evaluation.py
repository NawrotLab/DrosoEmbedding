# run_evaluation.py
import os
import pickle
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.utils.config_loader import load_config
from src.data.dataset import CustomDataset
from src.models.model_io import load_model
from src.models.cnn_transformer import CNN_Transformer
from src.utils.logger import setup_logger
from src.utils.helpers import get_latent_space, get_predictions, compute_tsne
from src.utils.analysis import (
    cosine_intra_class, cosine_inter_class,
    euclidean_intra_class, euclidean_inter_class,
    silhouette_scores
)


def evaluate_model(run_id, config_path, logger) -> dict:
    # Load config from pickle file
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
   
    # Load parameters and paths
    model_params = config["model"]["parameters"]
    training_params = config["training"]
    paths = config["paths"]
    eval_dir = paths["evaluation"]
    

    # Load test data
    with open(paths["pickle_path"], 'rb') as f:
        _, _, X_test, _, _, Y_test = pickle.load(f)

    # DataLoader
    test_dataset = CustomDataset(
        X_test, Y_test, transform=True,
        seq_length=model_params['seq_len'],
        seq_steps=model_params['seq_steps'],
        allTs_path=paths['allTs_path']
    )
    test_loader = DataLoader(test_dataset,
                             batch_size=training_params['batch_size'],
                             shuffle=False, num_workers=4, pin_memory=True)
    # temporary and dirty
    paths['models'] = f"{config['paths']['root']}/results/{config['data']['task']}_{run_id}/models/"
    paths['evaluation'] = f"{config['paths']['root']}/results/{config['data']['task']}_{run_id}/evaluation/"
    eval_dir = paths['evaluation']
    os.makedirs(eval_dir, exist_ok=True)

    logger.info(f"Locating Model at {paths['models']}")
    logger.info(f"Loading Model with parameters: {model_params} in {config['device']}.")
    # Load model
    classifier, _, train_loss, val_loss = load_model(
        CNN_Transformer, model_params, paths['models'], config['device']
    )
    if classifier is None:
        raise FileNotFoundError("Trained model not found.")

    # Inference
    # Get both latent spaces
    cnn_latent_space, latent_labels = get_latent_space(classifier, test_loader, config['device'], return_cnn_latent=True)
    transformer_latent_space, latent_labels = get_latent_space(classifier, test_loader, config['device'], return_cnn_latent=False)
    y_pred, y_true = get_predictions(classifier, test_loader, config['device'])


    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred,
                                        target_names=config['data']['classes'],
                                        output_dict=True)
    report = classification_report(y_true, y_pred, target_names=config['data']['classes'])
    with open(os.path.join(eval_dir, "ClassificationReport.txt"), 'w') as f:
        f.write(report)

    # Similarity metrics
    cos_intra = cosine_intra_class(transformer_latent_space, latent_labels)
    cos_inter, cos_labels = cosine_inter_class(transformer_latent_space, latent_labels)

    euc_intra = euclidean_intra_class(transformer_latent_space, latent_labels)
    euc_inter, euc_labels = euclidean_inter_class(transformer_latent_space, latent_labels)

    sil_score = silhouette_scores(transformer_latent_space, latent_labels)
    
    # after latent_space, latent_labels have been computed…
    tsne_2d = compute_tsne(transformer_latent_space,perplexity=config.get('plotting', {}).get('tsne_perplexity', 30))
    # Aggregate results
    results = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        
        'y_pred': y_pred, 
        'y_true': y_true,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report_dict': report_dict,
        
        'cnn_latent_space': cnn_latent_space,
        'transformer_latent_space': transformer_latent_space,
        'latent_labels': latent_labels,
        'tsne_2d': tsne_2d,

        'cosine_intra': cos_intra,
        'cosine_inter': cos_inter,
        'cosine_inter_labels': cos_labels,
        'euclidean_intra': euc_intra,
        'euclidean_inter': euc_inter,
        'euclidean_inter_labels': euc_labels,
        'silhouette_score': sil_score
    }

    # Save results
    out_path = os.path.join(eval_dir, f'{run_id}_evalResults.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)

    return results


def main():
    config = load_config()
    run_id = config['run_id']
    
    # Load config from pickle file
    # config_path = f"{config["paths"]["results_root"]}/{config['run_id']}/config_{config['run_id']}.pkl"
    config_path = f"{config["paths"]["results_root"]}/config.pkl"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}. Make sure to train the model first.")
        
    logger = setup_logger(task_name=config['run_id'], log_dir=os.path.join('logs/evaluation'))

    results = evaluate_model(run_id, config_path, logger)
    logger.info(f"Saved evaluation results to {config['paths']['evaluation']}/evaluation_results.pkl")

if __name__ == '__main__':
    main()

