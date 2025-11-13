from shiroinu.config import Config
from shiroinu.logger import Logger
from shiroinu.criteria import BaseLoss
from shiroinu.models.base_model import BaseModel
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


def get_conf_and_logger(conf_file, clear_logs):
    conf = Config.from_conf_file(conf_file)
    logger = Logger(conf.log_dir, clear_logs)
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


def create_instance(path, params, dataset_train=None, dataset_valid=None):
    class_ = load_class(path)

    dataset = None
    if issubclass(class_, BaseModel):
        dataset = dataset_train if (dataset_train is not None) else type(
            'TSDatasetDummy',
            (object,),
            {
              hyperparam[:-1]: [0.0] * len(getattr(dataset_valid, hyperparam[:-1]))
              for hyperparam in class_.data_based_hyperparams
            },
        )
    if issubclass(class_, BaseLoss):
        dataset = dataset_valid

    if 'means_' in class_.data_based_hyperparams:
        params['means_'] = dataset.means
    if 'stds_' in class_.data_based_hyperparams:
        params['stds_'] = dataset.stds
    if 'q1s_' in class_.data_based_hyperparams:
        params['q1s_'] = dataset.q1s
    if 'q2s_' in class_.data_based_hyperparams:
        params['q2s_'] = dataset.q2s
    if 'q3s_' in class_.data_based_hyperparams:
        params['q3s_'] = dataset.q3s

    return class_.create(**params)
