import os
import sys
import time
import pickle
import argparse
import logging
import json
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn 
import mlflow
import mlflow.pytorch

from src.utils.logger import setup_logger
from src.utils.config_loader import load_config
from src.data.dataset import CustomDataset
from src.models.cnn_transformer import CNN_Transformer
from src.training.training_routine import train_seq_seq_Classifier
from src.models.model_io import load_model, save_model


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, help="Unique run identifier")
if any('pydev' in arg for arg in sys.argv) or 'PYCHARM_HOSTED' in os.environ:
    args = parser.parse_args(args=[])
    args.run_name = "local_run"
else:
    args = parser.parse_args()


def main(logger):
    config = load_config()
    model_params = config["model"]["parameters"]
    time_start = time.time()

    logger.info(f"Process ID: {os.getpid()}")
    logger.info(f"Device: {config['device']}")

    mlflow.set_tracking_uri("file:/projects/group-share/MLflow/DrosoEmbedding")
    mlflow.set_experiment("DrosoEmbedding Experiments")

    with mlflow.start_run(run_name=config["data"]["task"]):
        try:
            # Log config parameters
            for section, params in config.items():
                if isinstance(params, dict):
                    for key, value in params.items():
                        if isinstance(value, (int, float, str, bool)):
                            mlflow.log_param(f"{section}_{key}", value)

            # Setup output paths
            out_root = config["paths"]["results_root"] + f"_{args.run_name}" if args.run_name else config["paths"]["results_root"]
            outPath_model = f"{out_root}/models"
            out_evaluation = f"{out_root}/evaluation"
            os.makedirs(outPath_model, exist_ok=True)
            os.makedirs(out_evaluation, exist_ok=True)

            # Load data
            with open(config["paths"]["pickle_path"], 'rb') as file:
                X_train, X_val, X_test, Y_train, Y_val, Y_test = pickle.load(file)
            logger.info(f"Loaded pickle: {config['data']['pickle_id']}, Sizes — Train: {len(Y_train)}, Val: {len(Y_val)}, Test: {len(Y_test)}")
            logger.info("============================================================")
            # Datasets and loaders
            train_dataset = CustomDataset(X_train, Y_train, transform=True,
                                        seq_length=config['data']['sequence']["length"],
                                        seq_steps=config['data']['sequence']["steps"],
                                          allTs_path=config["paths"]["allTs_path"],
                                          augment=False, num_augmentations=2)
            val_dataset = CustomDataset(X_val, Y_val, transform=True,
                                        seq_length=config['data']['sequence']["length"],
                                        seq_steps=config['data']['sequence']["steps"],
                                        allTs_path=config["paths"]["allTs_path"])

            train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"],
                                      shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
            val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"] * 2,
                                    shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

            # Load or initialize model
            Classifier, start_epoch, prev_train_loss, prev_val_loss = load_model(CNN_Transformer,
                                                                                  model_params,
                                                                                  outPath_model,
                                                                                  config["device"])
            criterion_name = config["training"]["criterion"]
            criterion_cls = getattr(nn, criterion_name)


            if Classifier is None:
                logger.info("config.data")
                for k, v in config["data"].items():
                    logger.info(f"{k}: {v}")
                logger.info("config.model.params")
                for k, v in config["model"]["parameters"].items():
                    logger.info(f"{k}: {v}")
                logger.info("config.training")
                for k, v in config["training"].items():
                    logger.info(f"{k}: {v}")
                logger.info("============================================================")



                Classifier = CNN_Transformer(**model_params)
                Classifier, training_loss, validation_loss = train_seq_seq_Classifier(
                    model=Classifier,
                    device=config["device"],
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=config['training']['epochs'],
                    lr=config['training']['learning_rate'],
                    weight_decay=config['training']['weight_decay'],
                    criterion=criterion_cls(),
                    logger=logger  
                )

                save_model(model=Classifier,
                           epoch=start_epoch + config['training']['epochs'],
                           model_class=CNN_Transformer,
                           params=model_params,
                           output_path=outPath_model,
                           train_loss=training_loss,
                           val_loss=validation_loss)

            # Timing and logging
            time_end = time.time() - time_start
            logger.info(f"Training took {time_end:.2f} seconds.")

            mlflow.log_metric("final_train_loss", training_loss[-1])
            mlflow.log_metric("final_val_loss", validation_loss[-1])
            mlflow.log_metric("training_time_sec", time_end)

            final_model_path = os.path.join(outPath_model, "final_model.pt")
            if os.path.exists(final_model_path):
                mlflow.log_artifact(final_model_path)

            log_file = os.path.join("logs/training", f"{config['run_id']}_{os.getenv('SLURM_JOB_ID', 'local')}.log")
            if os.path.exists(log_file):
                mlflow.log_artifact(log_file)

        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            mlflow.log_param("crashed", True)


if __name__ == "__main__":
    config = load_config()
    logger = setup_logger(task_name=config["run_id"], log_dir="logs/training")
    main(logger)
