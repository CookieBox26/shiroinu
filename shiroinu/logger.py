import torch
import os
import toml


class Logger:
    def __init__(self, log_dir):
        # if os.path.isdir(log_dir):
        #     raise ValueError(f'Log directory already exists: {log_dir}')
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.i_task = -1
        self.print_epoch = False

    def log(self, *args):
        self.log_file.write(' '.join([str(v) for v in args]) + '\n')
        self.log_file.flush()

    def start_task(self):
        self.i_task += 1
        print(f'===== task {self.i_task} =====')
        log_path = os.path.join(self.log_dir, f'log_task_{self.i_task}.txt')
        self.log_file = open(log_path, mode='w', encoding='utf-8', newline='\n')
        self.info = {'epochs': []}
        self.i_epoch = -1

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
        model_path = os.path.join(self.log_dir, f'model_task_{self.i_task}{suffix}.pth')
        torch.save(model.state_dict(), model_path)
        self.info[f'epoch_id{suffix}'] = self.i_epoch
        if self.print_epoch and (suffix == '_best'):
            print('loss_valid_best', self.info['epochs'][-1]['loss_0_per_sample_valid'])

    def end_task(self):
        self.log_file.close()
        info_path = os.path.join(self.log_dir, f'info_task_{self.i_task}.toml')
        if len(self.info['epochs']) > 0:
            with open(info_path, mode='w', encoding='utf8', newline='\n') as ofile:
                toml.dump(self.info, ofile)
            epoch_id_best = self.info['epoch_id_best']
            print(f'{epoch_id_best=}')
            print(self.info['epochs'][epoch_id_best]['loss_0_per_sample_valid'])
