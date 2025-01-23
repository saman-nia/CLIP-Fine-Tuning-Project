# we use utils.py for logging and making sure the directories exist and versiion tagging

import logging
import os
from datetime import datetime

def setup_logger(name="clip_trainer", level=logging.INFO):
    """
    this is a simple logger.
    in a real app, we can integrate it to logging and monitoring platform.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fnm = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        ch.setFormatter(fnm)
        logger.addHandler(ch)
    return logger

def ensure_dir_exists(dir_path):

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def generate_version_tag():
    """
    Generate a timestamp for each version for the latest traiend model.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

