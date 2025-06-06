import logging
import os
from datetime import datetime
import socket  


def setup_logger(task_name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Sets up a logger for a specific task (e.g., training, evaluation).

    Args:
        task_name (str): Name of the task (e.g., "train_model", "evaluate_model").
        log_dir (str): Directory to store log files.

    Returns:
        logging.Logger: Configured logger instance.
    """


    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    # Include timestamp and/or SLURM job ID in log filename
    timestamp = datetime.now().strftime("%Y%m%d") # was %Y%m%d_%H%M%S
    job_id = os.getenv("SLURM_JOB_ID", "local")  # Default to 'local' if not using SLURM
    log_file = os.path.join(log_dir, f"{timestamp}_{job_id}_{task_name}.log")
    

    # Configure logger
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # *newLine --> prevent duplicated logging in parent loggers

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Stream handler (console)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(levelname)s: %(message)s")
    stream_handler.setFormatter(stream_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


    # --- Log header metadata ---
    logger.info("=" * 60)
    logger.info(f"Logger initialized for task: {task_name}")
    logger.info(f"SLURM_JOB_ID: {job_id}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Hostname: {socket.gethostname()}")
    logger.info("=" * 60)

    return logger
