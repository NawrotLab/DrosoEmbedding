import os
import pickle
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.utils.config_loader import load_config
from src.data.dataset import CustomDataset
from src.models.model_io import load_model
from src.visualization.visualize_preformance import plot_confusion_matrix, plot_tsne, plot_train_val_loss,  plot_classes_cam, plot_umap, plot_metric_distribution, plot_similarity_matrix
from src.utils.logger import setup_logger
from src.models.cnn_transformer import CNN_Transformer
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from src.utils.helpers import get_latent_space, get_predictions
from src.utils.analysis import cosine_intra_class, cosine_inter_class, euclidean_intra_class, euclidean_inter_class, silhouette_scores

def evaluate_model(plot_visualizations: bool = True) -> dict:
    config = load_config()
    model_params = config["model"]["parameters"]
    training_params = config["training"]

    paths = config["paths"]
    eval_dir = paths["evaluation"]
    viz_dir = paths['visualizations']
    os.makedirs(eval_dir, exist_ok=True)
    # Load test data
    data_path = paths["pickle_path"]
    logger.info(f"Loading test data from {data_path}")
    try:
        with open(data_path, 'rb') as file:
            _, _, X_test, _, _, Y_test = pickle.load(file)
    except FileNotFoundError:
        logger.error("Test data file not found. Please ensure the file exists.")
        raise
    except Exception as e:
        logger.error(f"Failed to load test data: {e}", exc_info=True)
        raise

    logger.info(f"Test dataset loaded: {len(Y_test)} samples.")

    logger.info(f"all paths: {paths.items()}")

    # Prepare test dataset and DataLoader
    test_dataset = CustomDataset(
        X_test,
        Y_test,
        transform=True,
        seq_length= model_params['seq_len'],
        seq_steps = model_params['seq_steps'],
        allTs_path=paths['allTs_path']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_params['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Load model
    logger.info("Initializing and loading the model...")
    device = config['device']
    Classifier, _, train_loss, val_loss = load_model(
        model_class=CNN_Transformer,
        params=model_params,
        output_path=paths['models'],
        device=device
    )

    if Classifier is None:
        raise ValueError("Model not found. Please train the model first.")



    logger.info("Starting standard evaluation on test data...")
    latent_space, latent_labels = get_latent_space(Classifier, test_loader, device)
    logger.info(f"latent space shape: {latent_space.shape}")
    y_pred, y_true = get_predictions(Classifier, test_loader, device)



    # Compute evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=config['data']['classes'])
    logger.info(f"Classification Report: {report}")
    # avg_test_loss = total_test_loss / len(test_loader)

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    # logger.info(f"Test Loss: {avg_test_loss:.4f}")
    #logger.info("\nClassification Report:\n" + report)

    with open(f'{paths["evaluation"]}ClassificationReport.txt', 'w') as file:
        file.write(report)

    # Generate standard evaluation visualizations
    if plot_visualizations:
        os.makedirs(viz_dir, exist_ok=True)

        plot_train_val_loss(
            training_loss=train_loss,
            validation_loss=val_loss,
            dataID=paths['pickle_path'],
            model_name=Classifier.__class__.__name__,
            batch_size=config['training']['batch_size'],
            learning_rate=config['training']['learning_rate'],
            output_path=viz_dir
        )

        plot_confusion_matrix(
            cl_name="Evaluation",
            cm=cm,
            class_names=config['data']['classes'],
            output_path=viz_dir,
            dataID=paths['pickle_path'],
            hyperparameters={
                'batch_size': training_params['batch_size'],
                'seq_length': model_params['seq_len'],
                'embed_dim': model_params['embed_dim']
            }
        )

        plot_tsne(
            model=Classifier,
            data_loader=test_loader,
            device=config['device'],
            cl_name="Evaluation",
            output_path=viz_dir,
            dataID=paths['pickle_path'],
            class_names=config['data']['classes'],
            hyperparameters={
                'batch_size': training_params['batch_size'],
                'seq_length': model_params['seq_len'],
                'embed_dim': model_params['embed_dim']
            }
        )
        
        plot_umap(
            model=Classifier,
            data_loader=test_loader,
            device=config['device'],
            cl_name="Evaluation",
            output_path=viz_dir,
            dataID=paths['pickle_path'],
            class_names=config['data']['classes'],
            hyperparameters={
                'batch_size': training_params['batch_size'],
                'seq_length': model_params['seq_len'],
                'embed_dim': model_params['embed_dim']
            }
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


        # Why these no work all of the sudden? 
        # cams = {"GradCAM": GradCAM, "GradCAMPlusPlus": GradCAMPlusPlus, "ScoreCAM": ScoreCAM, "AblationCAM": AblationCAM,"HiResCAM": HiResCAM,"XGradCAM": XGradCAM,"EigenCAM": EigenCAM, "FullGrad": FullGrad}
        # plot_classes_cam(Classifier, device, test_loader, config['data']['classes'], viz_dir, cam_method=cams[config['visualization']['cam']["method"]], classifier_target_layer=config['visualization']['cam']["target_layer"])


        # visualize_full_sequences(data_loader=test_loader,
        #                          class_names=config["class_names"],
        #                          device=device,
        #                          cl_name='',
        #                          output_path=viz_dir,
        #                          dataID="Classes",
        #                          num_classes=model_params['nr_classes'])



    # Save standard evaluation results
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }

    return results



if __name__ == "__main__":
    config = load_config()
    logger = setup_logger(task_name=config["run_id"], log_dir="logs/evaluation")
    standard_results = evaluate_model(plot_visualizations=True)
    logger.info('Done')