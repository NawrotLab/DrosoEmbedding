from src.utils.logger import setup_logger
import logging
import time
from src.utils.config import training_config
import os
import pickle
from src.data.dataset import CustomDataset
from torch.utils.data import DataLoader
import torch
from src.models.cnn_transformer import CNN_Transformer
# from src.models.model_io import load_model
# from experiment.training_evaluation import train_LSTMClassier
from src.training.training_routine import train_seq_seq_Classifier
from src.models.model_io import load_model, save_model
from torch.optim import Adam
import argparse
import sys

# parser = argparse.ArgumentParser()
# parser.add_argument('--run_name', type=str, required=True, help="Unique run identifier")
# args = parser.parse_args()
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, help="Unique run identifier")
if any('pydev' in arg for arg in sys.argv) or 'PYCHARM_HOSTED' in os.environ:
    # Running in PyCharm or interactively, set a default run_name
    args = parser.parse_args(args=[])
    args.run_name = "local_run"
else:
    # Running via SLURM or command line
    args = parser.parse_args()


def main(logger):
    time_start = time.time()
    logger.info(f"Process ID: {os.getpid()}")
    try:

        logger.info(f"Ids, Parameters and Paths used:\n" + "\n".join(
            f"{key}: {value}" for key, value in training_config.items()))

        model_name = training_config["model_name"]
        run_name = training_config["run_name"]
        model_params = training_config["model_params"]

        # ________________________________________________________________________________________________________________________
        if args.run_name:
            out_root = f'{training_config["ROOT_PATH"]}/results/{model_name}_{run_name}_{args.run_name}'
        else:
            out_root = f'{training_config["ROOT_PATH"]}/results/{model_name}_{run_name}'

        outPath_model = f'{out_root}/models'
        out_evaluation = f'{out_root}/evaluation'
        os.makedirs(outPath_model, exist_ok=True)
        os.makedirs(out_evaluation, exist_ok=True)

        # -----------------------------------------------------------
        with open(training_config["data_PicklePath"], 'rb') as file:
            X_train, X_val, X_test, Y_train, Y_val, Y_test = pickle.load(file)
        logger.info(
            f'Pickle file: {training_config["pickleID_logTs"]} containing train, val, and test image paths and labels loaded')
        logger.info(f'Training Size: {len(Y_train)}; Val Size: {len(Y_val)}; Test_Size: {len(Y_test)}')

        train_dataset = CustomDataset(X_train, Y_train,
                                      transform=True,
                                      # embedding='AE',
                                      seq_length=model_params["seq_len"],
                                      seq_steps=model_params["seq_steps"],
                                      allTs_path=training_config["data_path_allTs"],
                                      augment=False,
                                      num_augmentations=2)
        train_loader = DataLoader(train_dataset,
                                  batch_size=training_config["batch_size"],
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True,
                                  persistent_workers=True)
        val_dataset = CustomDataset(X_val, Y_val,
                                    transform=True,
                                    # embedding='AE',
                                    seq_length=model_params["seq_len"],
                                    seq_steps=model_params["seq_steps"],
                                    allTs_path=training_config["data_path_allTs"])
        val_loader = DataLoader(val_dataset,
                                batch_size=training_config["batch_size"] * 2,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=True,
                                persistent_workers=True)


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f'Device: {device}')

        Classifier, start_epoch, prev_train_loss, prev_val_loss = load_model(CNN_Transformer,
                                                                             training_config["model_params"],
                                                                             outPath_model, device)
        logger.info(f'CLS: {Classifier}')
        if Classifier is None:

            Classifier = CNN_Transformer(nr_channels=model_params["nr_channels"], embed_dim=model_params["embed_dim"],
                                         num_heads=model_params["num_heads"], num_layers=model_params["num_layers"],
                                         nr_classes=model_params["nr_classes"], seq_len=model_params["seq_len"])


            Classifier, training_loss, validation_loss = train_seq_seq_Classifier(model=Classifier,
                                                                                  # optimizer = Classifier.optimizer,
                                                                                  device=device,
                                                                                  train_loader=train_loader, val_loader=val_loader,
                                                                                  num_epochs=model_params['num_epochs'],
                                                                                  lr=model_params['lr'],
                                                                                  weight_decay=model_params['weight_decay'],
                                                                                  criterion=model_params['criterion'])

            save_model(
                model=Classifier,
                epoch=start_epoch + model_params['num_epochs'],
                model_class=CNN_Transformer,
                params=model_params,
                output_path=outPath_model,
                train_loss=training_loss,
                val_loss=validation_loss,
            )
    

        time_end = time.time() - time_start
        logger.info(f"Training took {time_end:.2f} seconds.")


    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    logger = setup_logger(task_name=training_config["run_name"], log_dir="logs/training")
    # logger = setup_logger(task_name="train_model", log_dir="logs/training")
    main(logger)