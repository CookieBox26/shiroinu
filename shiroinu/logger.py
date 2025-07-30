import torch
import numpy as np
import os
import toml
import time


class Logger:
    def __init__(self, log_dir):
        # if os.path.isdir(log_dir):
        #     raise ValueError(f'Log directory already exists: {log_dir}')
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.i_task = -1
        self.print_epoch = False
        self.d_epoch_id_best = {}

    def log(self, *args):
        self.log_file.write(' '.join([str(v) for v in args]) + '\n')
        self.log_file.flush()

    def skip_task(self):
        self.i_task += 1
        print(f'===== skipping task {self.i_task} =====')

    def start_task(self):
        self.i_task += 1
        print(f'===== task {self.i_task} =====')
        log_path = os.path.join(self.log_dir, f'log_task_{self.i_task}.txt')
        self.log_file = open(log_path, mode='w', encoding='utf-8', newline='\n')
        self.info = {'epochs': []}
        self.i_epoch = -1
        self.time_0 = time.perf_counter()

    def add_info(self, key, value):
        self.info[key] = value

    def start_epoch(self):
        self.i_epoch += 1
        self.info['epochs'].append({})
        if self.print_epoch:
            print(f'----- epoch {self.i_epoch} -----')

    def add_info_epoch(self, key, value):
        self.info['epochs'][-1][key] = value

    def save_model(self, model, suffix):
        model_path = os.path.join(self.log_dir, f'model{suffix}_task_{self.i_task}.pth')
        torch.save(model.state_dict(), model_path)
        self.info[f'epoch_id{suffix}'] = self.i_epoch
        if self.print_epoch and (suffix == '_best'):
            print('loss_valid_best', self.info['epochs'][-1]['loss_0_per_sample_valid'])

    def save_array(self, key, value):
        array_path = os.path.join(self.log_dir, f'{key}_task_{self.i_task}.npy')
        if isinstance(value, np.ndarray):
            np.save(array_path, value)
        else:
            np.save(array_path, value.clone().detach().cpu().numpy())

    def end_task(self):
        duration = time.perf_counter() - self.time_0
        print(f'duration of task {self.i_task}: {duration:.3f}s')

        self.log_file.close()
        info_path = os.path.join(self.log_dir, f'info_task_{self.i_task}.toml')
        with open(info_path, mode='w', encoding='utf8', newline='\n') as ofile:
            toml.dump(self.info, ofile)
        if 'epoch_id_best' in self.info:
            epoch_id_best = self.info['epoch_id_best']
            self.d_epoch_id_best[self.i_task] = epoch_id_best
            print(f'{epoch_id_best=}')
            print(self.info['epochs'][epoch_id_best]['loss_0_per_sample_valid'])
