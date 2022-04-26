import logging
from datetime import datetime
from typing import Dict
import os
import pathlib


def get_logger(main_config: Dict, dataset_name: str):
    logger_name = f'{datetime.now().strftime(main_config["logger_name"])}.log'
    logger_folder = main_config["logger_path"].format(dataset_name=dataset_name)
    pathlib.Path(logger_folder).mkdir(parents=True, exist_ok=True)

    logger_path = os.path.join(logger_folder, logger_name)
    logger = _get_logger(logger_name, logger_path)
    return logger, logger_name, logger_path


def _get_logger(logger_name, log_path, level=logging.INFO):
    logger = logging.getLogger(logger_name)  # global variance?
    formatter = logging.Formatter('%(asctime)s : %(message)s')

    fileHandler = logging.FileHandler(log_path, mode='w')
    fileHandler.setFormatter(formatter)  # `formatter` must be a logging.Formatter
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    return logger