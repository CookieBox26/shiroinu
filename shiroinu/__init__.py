from shiroinu.config import Config
from shiroinu.logger import Logger
from importlib import import_module
import random
import numpy as np
import torch
import shutil
import os


def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def get_conf_and_logger(conf_file):
    conf = Config.from_conf_file(conf_file)
    logger = Logger(conf.log_dir)
    shutil.copy(conf_file, os.path.join(conf.log_dir, 'conf.toml'))
    return conf, logger


def load_class(path):
    try:
        module_path, class_name = path.rsplit('.', 1)
        module = import_module(module_path)
        model_class = getattr(module, class_name)
    except (ImportError, AttributeError):
        raise ImportError(path)
    return model_class


def load_instance(path, params):
    model_class = load_class(path)
    return model_class(**params)


def create_instance(path, params, dataset):
    model_class = load_class(path)

    if 'means_' in model_class.data_based_hyperparams:
        params['means_'] = dataset.means
    if 'stds_' in model_class.data_based_hyperparams:
        params['stds_'] = dataset.stds

    return model_class.create(**params)
