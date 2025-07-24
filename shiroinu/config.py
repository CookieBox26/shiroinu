import toml
import os
import math
import datetime
from types import SimpleNamespace


class Config:
    @staticmethod
    def prod_if_list(x):
        return math.prod(x) if isinstance(x, list) else x

    @staticmethod
    def preprocess(d):
        if 'data' in d:
            d['data']['path'] = os.path.expanduser(d['data']['path'])
        for key, targets in [('data', ['seq_len', 'pred_len'])]:
            for target in targets:
                d[key][target] = Config.prod_if_list(d[key][target])
        for key, sub_key, targets in [
            ('model', 'params', ['seq_len', 'pred_len', 'period_len']),
            ('models_eval', 'params', ['seq_len', 'pred_len', 'period_len']),
        ]:
            if key not in d:
                continue
            if not isinstance(d[key], list):
                for target in targets:
                    if target not in d[key][sub_key]:
                        continue
                    d[key][sub_key][target] = Config.prod_if_list(d[key][sub_key][target])
                continue
            for i in range(len(d[key])):
                for target in targets:
                    if target not in d[key][i][sub_key]:
                        continue
                    d[key][i][sub_key][target] = Config.prod_if_list(d[key][i][sub_key][target])

    def __init__(self, d):
        if 'mode' not in d:
            raise ValueError(f'Task mode is not specified.')
        Config.preprocess(d)
        suffix = ''
        if d['suffix']:
            suffix = '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        for k, v in d.items():
            if k in ['tasks', 'tasks_eval']:
                for task in v:
                    for kk, vv in task.items():
                        if kk in ['optimizer', 'lr_scheduler', 'batch_sampler']:
                            task[kk] = SimpleNamespace(**vv)
                setattr(self, k, [SimpleNamespace(**task) for task in v])
            else:
                setattr(self, k, v)
        self.log_dir = f'outputs/{self.out_dir_name}_{self.mode}{suffix}/'

    @staticmethod
    def from_conf_file(conf_file, mode):
        d = toml.load(conf_file)
        d['mode'] = mode
        return Config(d)
