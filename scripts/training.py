import os
import sys
import time
import pickle
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn 
import mlflow
import random
import numpy as np
import torch
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config
from src.data.dataset import CustomDataset
from src.models.cnn_transformer import CNN_Transformer
from src.training.training_routine import train_seq_seq_Classifier
from src.models.model_io import load_model, save_model
from src.utils.mlflow_utils import log_params_recursive


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, help="Unique run identifier")
if any('pydev' in arg for arg in sys.argv) or 'PYCHARM_HOSTED' in os.environ:
    args = parser.parse_args(args=[])
    args.run_name = "local_run"
else:
    args = parser.parse_args()


def main(config, logger):
    # Set random seeds for reproducibility
    slurm_id = int(config['slurm_id'])  # Ensure slurm_id is an integer
    seed = int(config["seeds"][slurm_id])  # Now we can safely use it as an index
    
    # Set Python random seed
    random.seed(seed)
    # Set NumPy random seed
    np.random.seed(seed)
    # Set PyTorch random seeds
    torch.manual_seed(seed)
    logger.info(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Using seed: {seed} for SLURM ID: {slurm_id}")
    
    os.environ.pop("MLFLOW_RUN_ID", None)  

    model_params = config["model"]["parameters"]
    training_params = config["training"]
    time_start = time.time()

    logger.info(f"Process ID: {os.getpid()}")
    logger.info(f"Device: {config['device']}")
    logger.info(f"Slurm ID: {slurm_id}")

    mlflow.set_tracking_uri("file:/rhomes/aabdel/DrosoEmbedding/mlflow")
    mlflow.set_experiment("DrosoEmbedding Experiments")


    with mlflow.start_run(run_name=config["run_id"], nested=False) as run:
        try:    
            # logger.info("torch version: " + torch.__version__)
      
            mlflow.set_tag("run_name", config["run_id"])  # optional, for clarity in UI
            mlflow_id = run.info.run_id
            mlflow.log_param("mlflow_run_id", mlflow_id)
        
            # Log config parameters
            log_params_recursive(config)

            # Setup output paths
            # out_root = config["paths"]["results_root"] + f"_{args.run_name}" if args.run_name else config["paths"]["results_root"]
            out_root = f'{config["paths"]["results_root"]}_{slurm_id}'

            logger.info(f"out root... {out_root}")
            outPath_model = f"{out_root}/models"
            out_evaluation = f"{out_root}/evaluation"
            best_dir = os.path.join(outPath_model, "best")
            os.makedirs(best_dir, exist_ok=True)
            os.makedirs(outPath_model, exist_ok=True)
            os.makedirs(out_evaluation, exist_ok=True)

            # Save for later
            with open(os.path.join(out_root, "mlflow_run_id.txt"), "w") as f:
                f.write(mlflow_id)

            # Load data
            with open(config["paths"]["pickle_path"], 'rb') as file:
                X_train, X_val, X_test, Y_train, Y_val, Y_test = pickle.load(file)
            logger.info(f"Loaded pickle: {config['data']['pickle_id']}, Sizes — Train: {len(Y_train)}, Val: {len(Y_val)}, Test: {len(Y_test)}")
            logger.info(f"example train: {X_train[0]}, {Y_train[0]}")
            if config["training"]["shuffle_labels_naive"]:
                random.shuffle(Y_train)
                logger.warning("Shuffled Y_train — naive label shuffling baseline enabled.")
                indices = random.sample(range(len(X_train)), 10)
                for idx in indices:
                    logger.info(f"Sample {idx}: X = {X_train[idx]}, Y = {Y_train[idx]}")
            elif config["training"]["shuffle_labels_consistantly"]:
                logger.warning("Shuffling Y_train labels in a consistant manner")
                with open(config["paths"]["pickle_path_shuffled"], 'rb') as file:
                    X_train, _, _, Y_train, _, _ = pickle.load(file)
                indices = random.sample(range(len(X_train)), 10)
                for idx in indices:
                    logger.info(f"Sample {idx}: X = {X_train[idx]}, Y = {Y_train[idx]}")



            logger.info("============================================================")
            # Datasets and loaders
            train_dataset = CustomDataset(X_train, Y_train, transform=True,
                                        seq_length=config['data']['sequence']['seq_len'],
                                        seq_steps=config['data']['sequence']['seq_steps'],
                                        allTs_path=config["paths"]["allTs_path"],
                                        preload_to_ram=training_params['preload_to_ram'])
            val_dataset = CustomDataset(X_val, Y_val, transform=True,
                                        seq_length=config['data']['sequence']['seq_len'],
                                        seq_steps=config['data']['sequence']['seq_steps'],
                                        allTs_path=config["paths"]["allTs_path"],
                                        preload_to_ram=training_params['preload_to_ram'])
            train_loader = DataLoader(train_dataset, batch_size=training_params["batch_size"],
                                      shuffle=True, num_workers=training_params['num_workers'], pin_memory=True, persistent_workers=True)
            val_loader = DataLoader(val_dataset, batch_size=training_params["batch_size"] * 2,
                                    shuffle=False, num_workers=training_params['num_workers'], pin_memory=True, persistent_workers=True)

            # Load or initialize model
            Classifier, start_epoch, prev_train_loss, prev_val_loss, prev_train_acc, prev_val_acc = load_model(CNN_Transformer,
                                                                                  model_params,
                                                                                  outPath_model,
                                                                                  config["device"])
            criterion_name = config["training"]["criterion"]
            label_smoothing = config["training"]["label_smoothing"]
            criterion_cls = getattr(nn, criterion_name)

            if Classifier is None:
                # Initialize new model
                logger.info("Initializing new model...")
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


                # --- shape smoke test ---
                if len(train_loader) > 0:
                    Classifier_tmp = CNN_Transformer(**model_params).to(config["device"])
                    Classifier_tmp.eval()
                    xb, yb = next(iter(train_loader))  # expects (B, C, H, W) or (B, S, C, H, W)
                    xb = xb.to(config["device"])
                    with torch.no_grad():
                        _ = Classifier_tmp(xb, debug_shapes=True)  # prints shapes at each step


                # -----------------------------------------------

                Classifier = CNN_Transformer(**model_params)
                num_epochs = training_params['epochs']
                start_epoch = 0
                training_loss = []
                validation_loss = []
                train_acc = []
                val_acc = []

            else:
                # Continue training existing model
                logger.info(f"Continuing training from epoch {start_epoch}...")
                logger.info(f"Previous training loss:{len(prev_train_loss),prev_train_loss}")
                logger.info(f"Previous training loss: {prev_train_loss[-1]:.4f}")
                logger.info(f"Previous validation loss: {prev_val_loss[-1]:.4f}")
                num_epochs = training_params['continue_training']['until_epoch']
                logger.info(f"Will continue training until epoch: {num_epochs} ")
                training_loss = prev_train_loss
                validation_loss = prev_val_loss
                train_acc = prev_train_acc
                val_acc = prev_val_acc


            LossClass = getattr(nn, criterion_name)
            if criterion_name == "CrossEntropyLoss" and float(label_smoothing) > 0.0:
                criterion = LossClass(label_smoothing=float(label_smoothing))
            else:
                criterion = LossClass()

            params_total = sum(p.numel() for p in Classifier.parameters() if p.requires_grad)
            logger.info(f"Trainable parameters: {params_total:,}")

            Classifier, new_training_loss, new_validation_loss, new_train_acc, new_val_acc = train_seq_seq_Classifier(
                model=Classifier,
                device=config["device"],
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                lr=training_params['learning_rate'],
                weight_decay=training_params['weight_decay'],
                start_epoch=start_epoch,
                criterion=criterion,
                logger=logger,  

                # --- NEW ---
                save_best_from=training_params.get('save_best_from'),  # warmup before tracking
                best_output_path=best_dir,
                model_class=CNN_Transformer,
                params=model_params,
                monitor=training_params.get('monitor', 'val_acc'),          # 'val_acc' or 'val_loss'
                mode=training_params.get('mode', 'max'),                    # 'max' for acc, 'min' for loss
            )



            # Combine previous and new losses
            training_loss.extend(new_training_loss)
            validation_loss.extend(new_validation_loss)
            train_acc.extend(new_train_acc)
            val_acc.extend(new_val_acc)

            # Save model
            final_epoch = start_epoch + num_epochs
            save_model(model=Classifier,
                       epoch=final_epoch,
                       model_class=CNN_Transformer,
                       params=model_params,
                       output_path=outPath_model,
                       train_loss=training_loss,
                       val_loss=validation_loss, 
                       train_acc=train_acc,
                       val_acc=val_acc, 
                       logger=logger)
            

            # Timing and logging
            time_end = time.time() - time_start
            logger.info(f"Training took {time_end:.2f} seconds.")

            mlflow.log_metric("final_train_loss", training_loss[-1])
            mlflow.log_metric("final_val_loss", validation_loss[-1])
            mlflow.log_metric("final_train_acc", train_acc[-1])
            mlflow.log_metric("final_val_acc", val_acc[-1])
            
            mlflow.log_metric("training_time_sec", time_end)

            final_model_path = os.path.join(outPath_model, "final_model.pt")
            if os.path.exists(final_model_path):
                mlflow.log_artifact(final_model_path)

            log_file = os.path.join("logs/training", f"{config['run_id']}_{slurm_id}.log")
            if os.path.exists(log_file):
                mlflow.log_artifact(log_file)

        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            mlflow.log_param("crashed", True)


if __name__ == "__main__":

    config = load_config()
    slurm_id = os.getenv("SLURM_ARRAY_TASK_ID", 0)
    config["slurm_id"] = slurm_id
    config["run_id"] = f"{config['run_id']}_{slurm_id}"
    logger = setup_logger(task_name=config["run_id"], log_dir="logs/training")
    # Save config dictionary to pickle file
    # config_file = f"{config["paths"]["results_root"]}/{config['run_id']}/config_{config['run_id']}_{slurm_id}.pkl"
    config_file = f'{config['paths']['root']}/results/{config['data']['task']}_{config['run_id']}/config.pkl'
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, 'wb') as f:
        pickle.dump(config, f)
    logger.info(f"Saved config to {config_file}")
    
    main(config, logger)
