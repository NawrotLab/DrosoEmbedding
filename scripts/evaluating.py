import os
import pickle
import torch
import mlflow
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.utils.config_loader import load_config
from src.data.dataset import CustomDataset
from src.models.model_io import load_model
from src.models.cnn_transformer import CNN_Transformer
from src.utils.logger import setup_logger
from src.visualization.visualize_preformance import (
    plot_confusion_matrix, plot_tsne, plot_train_val_loss,
    plot_classes_cam, plot_umap, plot_metric_distribution, plot_similarity_matrix, plot_weighted_avg, plot_per_class_metrics, plot_umap_latent, plot_tsne_latent 
)
from src.utils.helpers import get_latent_space, get_predictions
from src.utils.analysis import (
    cosine_intra_class, cosine_inter_class,
    euclidean_intra_class, euclidean_inter_class,
    silhouette_scores
)


def evaluate_model(config, logger, plot_visualizations: bool = True) -> dict:
    model_params = config["model"]["parameters"]
    training_params = config["training"]
    paths = config["paths"]
    eval_dir = paths["evaluation"]
    viz_dir = paths["visualizations"]
    os.makedirs(eval_dir, exist_ok=True)

    # Load test data
    logger.info(f"Loading test data from {paths['pickle_path']}")
    with open(paths["pickle_path"], 'rb') as file:
        _, _, X_test, _, _, Y_test = pickle.load(file)
    logger.info(f"Test dataset loaded: {len(Y_test)} samples.")

    # Prepare DataLoader
    test_dataset = CustomDataset(
        X_test, Y_test, transform=True,
        seq_length=model_params['seq_len'],
        seq_steps=model_params['seq_steps'],
        allTs_path=paths['allTs_path']
    )
    test_loader = DataLoader(
        test_dataset, batch_size=training_params['batch_size'],
        shuffle=False, num_workers=4, pin_memory=True
    )

    # Load model
    logger.info("Loading trained model...")
    Classifier, _, train_loss, val_loss = load_model(
        CNN_Transformer, model_params, paths['models'], config['device']
    )
    if Classifier is None:
        raise ValueError("Model not found. Please train the model first.")

    # Evaluation
    logger.info("Running evaluation...")
    latent_space, latent_labels = get_latent_space(Classifier, test_loader, config['device'])
    y_pred, y_true = get_predictions(Classifier, test_loader, config['device'])

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=config['data']['classes'], output_dict=True)
    report = classification_report(y_true, y_pred, target_names=config['data']['classes'])

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Report:\n{report}")

    with open(os.path.join(eval_dir, "ClassificationReport.txt"), 'w') as f:
        f.write(report)

    if plot_visualizations:
        os.makedirs(viz_dir, exist_ok=True)

        plot_weighted_avg(report_dict, viz_dir)
        plot_per_class_metrics(report_dict, viz_dir)

        plot_train_val_loss(train_loss, val_loss, paths['pickle_path'],
                            Classifier.__class__.__name__,
                            training_params['batch_size'],
                            training_params['learning_rate'],
                            output_path=viz_dir)

        plot_confusion_matrix("Evaluation", cm, config['data']['classes'],
                              output_path=viz_dir, dataID=paths['pickle_path'],
                              hyperparameters={
                                  'batch_size': training_params['batch_size'],
                                  'seq_length': model_params['seq_len'],
                                  'embed_dim': model_params['embed_dim']
                              })

        # plot_tsne(Classifier, test_loader, config['device'], "Evaluation",
        #           output_path=viz_dir, dataID=paths['pickle_path'],
        #           class_names=config['data']['classes'],
        #           hyperparameters={
        #               'batch_size': training_params['batch_size'],
        #               'seq_length': model_params['seq_len'],
        #               'embed_dim': model_params['embed_dim']
        #           })

        # plot_umap(Classifier, test_loader, config['device'], "Evaluation",
        #           output_path=viz_dir, dataID=paths['pickle_path'],
        #           class_names=config['data']['classes'],
        #           hyperparameters={
        #               'batch_size': training_params['batch_size'],
        #               'seq_length': model_params['seq_len'],
        #               'embed_dim': model_params['embed_dim']
        #           })
        

                # 2a) TSNE
        fig_tsne, ax_tsne = plot_tsne_latent(
            latent_space,
            latent_labels,
            class_names=config['data']['classes'],
            output_path=viz_dir,
            filename=f"{len(config['data']['classes'])}Cls_TSNE2.png",
            title=f"{len(config['data']['classes'])}-way TSNE"
        )

        # 2b) UMAP
        fig_umap, ax_umap = plot_umap_latent(
            latent_space,
            latent_labels,
            class_names=config['data']['classes'],
            output_path=viz_dir,
            filename=f"{len(config['data']['classes'])}Cls_UMAP2.png",
            title=f"{len(config['data']['classes'])}-way UMAP"
        )
        


        # --- COSINE ---
        cos_intra = cosine_intra_class(latent_space, latent_labels)
        cos_inter, cos_labels = cosine_inter_class(latent_space, latent_labels)
        plot_metric_distribution(cos_intra, os.path.join(viz_dir, "cosine_intra_boxplot.png"), "Intra-class Cosine Similarity")
        plot_similarity_matrix(cos_inter, cos_labels, os.path.join(viz_dir, "cosine_inter_heatmap.png"), "Inter-class Cosine Similarity")

        # --- EUCLIDEAN ---
        euc_intra = euclidean_intra_class(latent_space, latent_labels)
        euc_inter, euc_labels = euclidean_inter_class(latent_space, latent_labels)
        plot_metric_distribution(euc_intra, os.path.join(viz_dir, "euclidean_intra_boxplot.png"), "Intra-class Euclidean Distance")
        plot_similarity_matrix(euc_inter, euc_labels, os.path.join(viz_dir, "euclidean_inter_heatmap.png"), "Inter-class Euclidean Distance")

        # --- SILHOUETTE ---
        sil_scores = silhouette_scores(latent_space, latent_labels)
        plot_metric_distribution(sil_scores, os.path.join(viz_dir, "silhouette_score_boxplot.png"), "Silhouette Score per Class")

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }


def main():
    config = load_config()
    logger = setup_logger(task_name=config["run_id"], log_dir="logs/evaluation")

    mlflow.set_tracking_uri("file:/projects/group-share/MLflow/DrosoEmbedding")
    mlflow.set_experiment("DrosoEmbedding Experiments")

    # Try to find saved mlflow_id from training
    mlflow_id_path = os.path.join(config["paths"]["results_root"], "mlflow_run_id.txt")
    use_existing_run = os.path.exists(mlflow_id_path)

    if use_existing_run:
        with open(mlflow_id_path, "r") as f:
            mlflow_id = f.read().strip()
        logger.info(f"Found saved mlflow ID, logging to existing run: {mlflow_id}")
    else:
        mlflow_id = None
        logger.info("No saved mlflow ID found, starting new run.")

    with mlflow.start_run(run_id=mlflow_id, run_name=config["run_id"]):
        mlflow.log_param("stage", "evaluation")

        try:
            results = evaluate_model(config, logger, plot_visualizations=True)
            mlflow.log_metric("test_accuracy", results['accuracy'])

            report_path = os.path.join(config["paths"]["evaluation"], "ClassificationReport.txt")
            if os.path.exists(report_path):
                mlflow.log_artifact(report_path)

            log_file = os.path.join("logs/evaluation", f"{config['run_id']}_{os.getenv('SLURM_JOB_ID', 'local')}.log")
            if os.path.exists(log_file):
                mlflow.log_artifact(log_file)

        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            mlflow.log_param("evaluation_crashed", True)


if __name__ == "__main__":
    main()