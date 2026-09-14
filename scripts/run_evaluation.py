# run_evaluation.py
import os
import pickle
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

from src.utils.config_loader import load_config
from src.data.dataset import CustomDataset
from src.models.model_io import load_model
from src.models.cnn_transformer import CNN_Transformer
from src.utils.logger import setup_logger
from src.utils.helpers import get_latent_space, get_predictions, compute_tsne
from src.visualization.visualize_performance import compute_class_mean_cams
from pytorch_grad_cam import GradCAM
from src.utils.helpers import paths2neuropilpaths


def evaluate_model(run_id, config, logger) -> dict:
    # Everything -- data paths, model architecture defaults, task/classes --
    # comes from the live config (load_config(), env-var driven), not a
    # frozen training-time snapshot. The one thing that must exactly match
    # how the model was trained (its architecture hyperparameters) is
    # recovered from the checkpoint itself inside load_model(), which is
    # self-describing (see model_io.py::save_model's 'params' field) -- so
    # no per-run config.pkl is needed for that either.
    model_params = config["model"]["parameters"]
    training_params = config["training"]
    paths = config["paths"]

    # Load test data
    with open(paths["pickle_path"], 'rb') as f:
        _, X_val, X_test, _, Y_val, Y_test = pickle.load(f)

    X_val  = paths2neuropilpaths(X_val, config)
    X_test = paths2neuropilpaths(X_test, config)

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
    logger.info(f"Test loader: {len(test_loader)}, Val loader: {len(val_loader)}")
    logger.info(f"Test example path: {X_test[0]}")

    # Published layout: checkpoints live flat under models/<task>/, and cached
    # evaluation results under evaluation/<task>/ (what figures read) --
    # not the old per-run results/{task}_{run_id}/... cluster layout.
    # Both independently overridable via DROSO_MODELS_DIR / DROSO_EVAL_DIR.
    paths['models'] = paths['models_dir']
    eval_dir = paths['eval_dir']
    os.makedirs(eval_dir, exist_ok=True)
    eval_file = os.path.join(eval_dir, f"{run_id}_evalResults.pkl")
    if os.path.exists(eval_file):
        logger.info(f"Evaluation results already exist at {eval_file}")
        with open(eval_file, 'rb') as f:
            return pickle.load(f)

    logger.info(f"Locating Model at {paths['models']}")
    logger.info(f"Loading Model with parameters: {model_params} in {config['device']}.")
    # Load model
    classifier, _, train_loss, val_loss, train_acc, val_acc = load_model(
        CNN_Transformer, model_params, paths['models'], config['device'],
        run_id=run_id, logger=logger
    )
    if classifier is None:
        raise FileNotFoundError("Trained model not found.")

    # Inference
    transformer_latent_space, latent_labels = get_latent_space(classifier, test_loader, config['device'], return_cnn_latent=False)
    logger.info(f"Transformer Latent Space shape: {transformer_latent_space.shape}")
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

    # Metrics -- both dict and human-readable text forms are kept inside
    # the results dict itself (not written as separate .txt files: nothing
    # reads those, and they were pure duplicates of classification_report_dict
    # anyway; this also gives the val-set report a dict form it never had).
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=config['data']['classes'], output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=config['data']['classes'])
    report_dict_val = classification_report(Y_true_VAL, Y_pred_VAL, target_names=config['data']['classes'], output_dict=True)
    report_text_val = classification_report(Y_true_VAL, Y_pred_VAL, target_names=config['data']['classes'])

    # after latent_space, latent_labels have been computed…
    tsne_2d = compute_tsne(transformer_latent_space, perplexity=30)
    logger.info(f"TSNE 2D shape: {tsne_2d.shape}")
    logger.info(f"TSNE 2D range: {np.max(tsne_2d), np.min(tsne_2d)}")


    # Aggregate results
    results = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,

        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report_dict': report_dict,
        'classification_report_text': report_text,
        'classification_report_dict_val': report_dict_val,
        'classification_report_text_val': report_text_val,

        'transformer_latent_space': transformer_latent_space,
        'latent_labels': latent_labels,
        'tsne_2d': tsne_2d,

        'mean_cams': mean_cams
    }

    with open(eval_file, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Saved evaluation results to {eval_file}")

    return results


def main():
    config = load_config()
    run_id = config['run_id']
    logger = setup_logger(task_name=config['run_id'], log_dir=os.path.join('logs/evaluation'))
    evaluate_model(run_id, config, logger)

if __name__ == '__main__':
    main()
