import os
import logging
from datetime import datetime
import json
from dataclasses import asdict, is_dataclass


def setup_logger(log_dir: str, log_name: str) -> logging.Logger:
    """
    Create and return a logger that logs to both file and stdout.
    
    Args:
        log_dir: Directory to save log files
        log_name: Base name of the log file (without timestamp)

    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent duplicate logs

    # Clear existing handlers (important in Jupyter / repeated runs)
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def log_config(logger: logging.Logger, config) -> None:
    """
    Log config parameters in a structured and readable JSON format.
    
    Args:
        logger: logger instance
        config: dataclass config object
    """
    if is_dataclass(config):
        config_dict = asdict(config)
    else:
        config_dict = config

    logger.info("========== Configuration ==========")
    logger.info("\n" + json.dumps(config_dict, indent=4))
    logger.info("===================================")