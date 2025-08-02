import toml
import os
import math
import datetime
from types import SimpleNamespace
import copy


class Config:
    @staticmethod
    def prod_if_list(x):
        return math.prod(x) if isinstance(x, list) else x

    def _set_data(self, d):
        d['data']['path'] = os.path.expanduser(d['data']['path'])
        for field in ['seq_len', 'pred_len']:
            d['data'][field] = Config.prod_if_list(d['data'][field])
        self.data = d['data']

    def _preprocess_model(self, model):
        for field in ['seq_len', 'pred_len', 'period_len']:
            if field in model['params']:
                model['params'][field] = Config.prod_if_list(model['params'][field])
        return model

    def _set_models(self, d):
        for i_model in range(len(d['models'])):
            d['models'][i_model] = self._preprocess_model(d['models'][i_model])
        self.models = d['models']

    def _set_tasks(self, d):
        for i_task in range(len(d['tasks'])):
            if 'task_template_id' in d['tasks'][i_task]:
                template = d['task_templates'][d['tasks'][i_task]['task_template_id']]
                for k, v in template.items():
                    if k not in d['tasks'][i_task]:
                        d['tasks'][i_task][k] = v
            for field in ['optimizer', 'lr_scheduler', 'batch_sampler']:
                if field in d['tasks'][i_task]:
                    d['tasks'][i_task][field] = SimpleNamespace(**d['tasks'][i_task][field])
        self.tasks = [SimpleNamespace(**task) for task in d['tasks']]

    def get_model(self, id, state_path=''):
        model_ = copy.deepcopy(self.models[id])
        if state_path != '':
            if '<HERE>' in state_path:
                state_path = state_path.replace('<HERE>', self.log_dir)
            model_['params']['state_path'] = state_path
        return model_

    def __init__(self, d):
        self.criteria = []

        fields = [
            'out_dir_name', 'batch_size_eval', 'data',
            'criteria', 'models', 'tasks',
        ]
        for field in fields:
            if field in ['out_dir_name', 'batch_size_eval', 'data']:
                assert field in d, f'A required field is missing: {field}'
            if field == 'data':
                self._set_data(d)
            elif field == 'models':
                self._set_models(d)
            elif field == 'tasks':
                self._set_tasks(d)
            else:
                setattr(self, field, d[field])

        suffix = '' if (('suffix' not in d) or not d['suffix']) \
            else '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        self.log_dir = f'outputs/{self.out_dir_name}{suffix}/'


    @staticmethod
    def from_conf_file(conf_file):
        d = toml.load(conf_file)
        if d['out_dir_name'] == '<CONF_FILE_NAME>':
            out_dir_name, _ = os.path.splitext(os.path.basename(conf_file))
            d['out_dir_name'] = out_dir_name
        return Config(d)
