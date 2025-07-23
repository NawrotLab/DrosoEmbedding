# run_plots.py
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.visualization.visualize_preformance import plot_confusion_matrix, plot_train_val_loss, plot_weighted_avg, plot_per_class_metrics, plot_tsne_latent, plot_metric_distribution, plot_similarity_matrix
from src.utils.helpers import compute_tsne

def plot_all(results: dict, config: dict, logger) -> None:
    """
    Generate all performance and latent-space visualizations from precomputed results.
    """
    viz_dir = config['paths']['visualizations']
    os.makedirs(viz_dir, exist_ok=True)
    model_params = config["model"]["parameters"]
    training_params = config["training"]
    paths = config["paths"]


    # Unpack results
    rd            = results['classification_report_dict']
    ts            = results['train_loss']
    vs            = results['val_loss']
    cm            = results['confusion_matrix']
    cnn_latent    = results['cnn_latent_space']
    transformer_latent = results['transformer_latent_space']
    ll            = results['latent_labels']
    tsne_2d       = results['tsne_2d']
    # umap_2d       = results['umap_2d']
    ci            = results['cosine_intra']
    co            = results['cosine_inter']
    cl            = results['cosine_inter_labels']
    ei            = results['euclidean_intra']
    eo            = results['euclidean_inter']
    el            = results['euclidean_inter_labels']
    ss            = results['silhouette_score']
    class_names   = config['data']['classes']

    # 1. Training/validation loss
    plot_train_val_loss(
        ts, vs,
        dataID='',
        model_name=config['model']['architecture'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        output_path=viz_dir,
        ylim=None
    )

    # 2. Classification summary
    plot_weighted_avg(rd, viz_dir)
    plot_per_class_metrics(rd, viz_dir)

    # 3. Confusion matrix
    plot_confusion_matrix("Evaluation", cm, config['data']['classes'],
                            output_path=viz_dir, dataID=paths['pickle_path'],
                            hyperparameters={
                                'batch_size': training_params['batch_size'],
                                'seq_length': model_params['seq_len'],
                                'embed_dim': model_params['transformer_embed_dim']
                            })

    # 4. Latent-space 2D projections for both CNN and Transformer
    # CNN Latent Space
    tsne_2d_cnn = compute_tsne(cnn_latent)
    
    plot_tsne_latent(
        latent_2d=tsne_2d_cnn,
        labels_np=ll,
        class_names=class_names,
        output_path=viz_dir,
        filename=f"{len(class_names)}Cls_CNN_TSNE2.png",
        title="t-SNE Projection (CNN Latent Space)"
    )
    
    # Transformer Latent Space
    plot_tsne_latent(
        latent_2d=tsne_2d,
        labels_np=ll,
        class_names=class_names,
        output_path=viz_dir,
        filename=f"{len(class_names)}Cls_Transformer_TSNE2.png",
        title="t-SNE Projection (Transformer Latent Space)"
    )

    # 5. Similarity analyses
    # Cosine
    plot_metric_distribution(
        ci,
        os.path.join(viz_dir, 'cosine_intra.png'),
        'Intra-class Cosine Similarity'
    )
    plot_similarity_matrix(
        matrix=co,
        class_labels=cl,
        output_path=os.path.join(viz_dir, 'cosine_inter.png'),
        title='Inter-class Cosine Similarity'
    )

    # Euclidean
    plot_metric_distribution(
        ei,
        os.path.join(viz_dir, 'euclidean_intra.png'),
        'Intra-class Euclidean Distance'
    )
    plot_similarity_matrix(
        matrix=eo,
        class_labels=el,
        output_path=os.path.join(viz_dir, 'euclidean_inter.png'),
        title='Inter-class Euclidean Distance'
    )

    # Silhouette
    plot_metric_distribution(
        ss,
        os.path.join(viz_dir, 'silhouette_score.png'),
        'Silhouette Score per Class'
    )



def main():
    # Load config and logger
    config = load_config()
    run_id = config['run_id']
    logger = setup_logger(
        task_name=config['run_id'],
        log_dir='logs/plots/'
    )

    # Load evaluation results
    eval_path = os.path.join(config['paths']['evaluation'], f'{run_id}_evalResults.pkl')
    if not os.path.exists(eval_path):
        logger.error(f"Evaluation results not found at {eval_path}")
        return

    with open(eval_path, 'rb') as f:
        results = pickle.load(f)

    # Generate plots
    plot_all(results, config, logger)
    logger.info(f"All plots saved to {config['paths']['visualizations']}")

if __name__ == '__main__':
    main()
