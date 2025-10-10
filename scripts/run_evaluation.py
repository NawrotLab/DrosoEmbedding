# run_evaluation.py
import os
import pickle
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

from src.utils.config_loader import load_config
from src.data.dataset import CustomDataset
from src.models.model_io import load_model
from src.models.cnn_transformer import CNN_Transformer
from src.utils.logger import setup_logger
from src.utils.helpers import get_latent_space, get_predictions, compute_tsne
from src.utils.analysis import (
    cosine_intra_class, cosine_inter_class,
    euclidean_intra_class, euclidean_inter_class,
    silhouette_scores, compute_centroid_tsne_payload
)
from src.visualization.visualize_preformance import plot_mean_cams, compute_class_mean_cams
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad



def evaluate_model(run_id, config_path, logger, out_dir) -> dict:
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
        _, X_val, X_test, _, Y_val, Y_test = pickle.load(f)

    # DataLoader
    test_dataset = CustomDataset(
        X_test, Y_test, transform=True,
        seq_length=model_params['seq_len'],
        seq_steps=model_params['seq_steps'],
        allTs_path=paths['allTs_path']
    )

    val_dataset = CustomDataset(
        X_val, Y_val, transform=True,
        seq_length=model_params['seq_len'],
        seq_steps=model_params['seq_steps'],
        allTs_path=paths['allTs_path']
    )

    test_loader = DataLoader(test_dataset,
                             batch_size=training_params['batch_size'],
                             shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=training_params['batch_size'],
                            shuffle=False, num_workers=4, pin_memory=True)
    # temporary and dirty
    paths['models'] = f"{config['paths']['root']}/results/{config['data']['task']}_{run_id}/models/best/"
    paths['evaluation'] = f"{config['paths']['root']}/results/{config['data']['task']}_{run_id}/evaluation/"
    eval_dir = paths["evaluation"]

    os.makedirs(eval_dir, exist_ok=True)

    logger.info(f"Locating Model at {paths['models']}")
    logger.info(f"Loading Model with parameters: {model_params} in {config['device']}.")
    # Load model
    classifier, _, train_loss, val_loss, train_acc, val_acc = load_model(
        CNN_Transformer, model_params, paths['models'], config['device'], logger
    )
    if classifier is None:
        raise FileNotFoundError("Trained model not found.")

    # Inference
    # Get both latent spaces
    cnn_latent_space, latent_labels = get_latent_space(classifier, test_loader, config['device'], return_cnn_latent=True)
    logger.info(f"CNN Latent Space shape: {cnn_latent_space.shape}")
    # logger.info(f"CNN Latent Space range: {np.max(cnn_latent_space), np.min(cnn_latent_space)}")
    transformer_latent_space, latent_labels = get_latent_space(classifier, test_loader, config['device'], return_cnn_latent=False)
    logger.info(f"Transformer Latent Space shape: {transformer_latent_space.shape}")
    # logger.info(f"Transformer Latent Space range: {np.max(transformer_latent_space), np.min(transformer_latent_space)}")
    y_pred, y_true = get_predictions(classifier, test_loader, config['device'])
    Y_pred_VAL, Y_true_VAL = get_predictions(classifier, val_loader, config['device'])

    # CAMs
    mean_cams, _ = compute_class_mean_cams(
    classifier,
    test_loader,                 # your labeled eval loader
    config['data']['classes'],
    config['device'],
    cam_method=GradCAM,         # or other CAM method from the lib
    target_layer_index=3,       # adapt to your model.cnn depth
    use_reshape_transform=True, # True if you might have 5D features
    max_items_per_class=None,     # reasonable cap; set None to use all
    return_all=False)

    # for cls, arr in mean_cams.items():
    #     logger.info(f"{cls}: {arr.shape}, dtype={arr.dtype}, min={arr.min():.3f}, max={arr.max():.3f}")
    
    # logger.info(f"Computed mean CAMs. Shapes: {mean_cams.values()}")

    # fig = plot_mean_cams(mean_cams, cols=4, figsize=(10, 8), suptitle="Mean Grad-CAM (Layer 3)", add_colorbar=True, save_path="Mean_GradCAM_L3.png")

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cm_val = confusion_matrix(Y_true_VAL, Y_pred_VAL)
    report_dict = classification_report(y_true, y_pred, target_names=config['data']['classes'], output_dict=True)
    report = classification_report(y_true, y_pred, target_names=config['data']['classes'])
    report_val = classification_report(Y_true_VAL, Y_pred_VAL, target_names=config['data']['classes'])
    with open(os.path.join(eval_dir, "ClassificationReport.txt"), 'w') as f:
        f.write(report)
    with open(os.path.join(eval_dir, "ClassificationReport_VAL.txt"), 'w') as f:
        f.write(report_val)

    # Similarity metrics
    cos_intra = cosine_intra_class(transformer_latent_space, latent_labels)
    cos_inter, cos_labels = cosine_inter_class(transformer_latent_space, latent_labels)
    

    euc_intra = euclidean_intra_class(transformer_latent_space, latent_labels)
    euc_inter, euc_labels = euclidean_inter_class(transformer_latent_space, latent_labels)

    sil_score = silhouette_scores(transformer_latent_space, latent_labels)


    # --- NEW: grouped centroid -> t-SNE (ready-to-plot) ---
    try:
        payload = compute_centroid_tsne_payload(
            features_hd = transformer_latent_space,          # (N, D)
            labels      = latent_labels,                     # (N,)
            task_name   = config['data']['task'],            # 'MetabolicState_2' / 'State_Modality_6' / 'State_Modality_Valence_16'
            class_names = config['data']['classes'],         # list[str] aligned to labels
            # random_state=30  # optional override
        )
        centroid_names = payload['centroid_group_names']
        centroid_Z     = payload['centroid_tsne2d']
        logger.info(f"Centroid t-SNE: groups={centroid_names} shape={centroid_Z.shape}")
    except Exception as e:
        logger.error(f"Centroid t-SNE computation failed: {e}")
        centroid_names, centroid_Z = None, None

    

    
    # after latent_space, latent_labels have been computed…
    tsne_2d = compute_tsne(transformer_latent_space, perplexity=500)
    logger.info(f"TSNE 2D shape: {tsne_2d.shape}")
    logger.info(f"TSNE 2D range: {np.max(tsne_2d), np.min(tsne_2d)}")
    
    
    # Aggregate results
    results = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        
        'y_pred': y_pred, 
        'y_true': y_true,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report_dict': report_dict,
        
        'cnn_latent_space': cnn_latent_space,
        'transformer_latent_space': transformer_latent_space,
        'latent_labels': latent_labels,
        'tsne_2d': tsne_2d,

        'centroid_group_names': centroid_names,   # None if failed
        'centroid_tsne2d': centroid_Z,            # None if failed

        'mean_cams': mean_cams,

        'cosine_intra': cos_intra,
        'cosine_inter': cos_inter,
        'cosine_inter_labels': cos_labels,
        'euclidean_intra': euc_intra,
        'euclidean_inter': euc_inter,
        'euclidean_inter_labels': euc_labels,
        'silhouette_score': sil_score
    }

    # logger.info(f"results {results}")
    # Save results
    # out_path = os.path.join(eval_dir, f'{run_id}_evalResults.pkl')
    with open(out_dir, 'wb') as f:
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
    eval_dir = f"{config['paths']['root']}/results/{config['data']['task']}_{run_id}/evaluation/{run_id}_tsnep500_evalResults.pkl"
    if not os.path.exists(eval_dir):
        evaluate_model(run_id, config_path, logger, eval_dir)
        logger.info(f"Saved evaluation results to {eval_dir}")
    else:
        logger.info(f"Evaluation results already exist at {eval_dir}")

if __name__ == '__main__':
    main()

