import collections
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import gc


class TSDataset(Dataset):
    TSBatch = collections.namedtuple('TSBatch', [
        'tsta', 'tste', 'data', 'datass',
        'tsta_future', 'tste_future', 'data_future', 'datass_future',
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def to_tensor(cls, x):
        return torch.tensor(x, dtype=torch.float32, device=cls.device)

    def __init__(self, logger, df, seq_len, horizon, cols_org):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.df = df
        self.tsta = list(self.df['timestamp'].values)
        self.tste = list(self.df['timestep'].values)
        del self.df['timestamp'], self.df['timestep']
        self.seq_len = seq_len
        self.horizon = horizon
        self.cols_org = cols_org

        self.n_sample = len(df) - (seq_len - 1) - horizon
        if logger is not None:
            logger.log(f'===== {self.__class__.__name__} instantiated. =====')
            logger.log(str(list(df.columns)))
            logger.log(f'{len(df)=}')
            logger.log(f'{self.n_sample=}')
            logger.log(f'{self.tsta[0                        ]=}')
            logger.log(f'{self.tsta[0 + seq_len - 1          ]=}')
            logger.log(f'{self.tsta[0 + seq_len              ]=}')
            logger.log(f'{self.tsta[0 + seq_len + horizon - 1]=}')
            logger.log(f'{self.tsta[self.n_sample - 1                        ]=}')
            logger.log(f'{self.tsta[self.n_sample - 1 + seq_len - 1          ]=}')
            logger.log(f'{self.tsta[self.n_sample - 1 + seq_len              ]=}')
            logger.log(f'{self.tsta[self.n_sample - 1 + seq_len + horizon - 1]=}')

        self.n_feats = len(self.df.columns)
        self.means = []
        self.stds = []
        for col in self.df.columns:
            self.means.append(self.df[col].mean())
            self.stds.append(self.df[col].std())

    def get_info_for_logger(self):
        return {
            'cols': self.df.columns,
            'cols_org': self.cols_org,
            'steps': len(self.df),
            'n_sample': self.n_sample,
            'sample_first': [
                self.tsta[0],
                self.tsta[0 + self.seq_len - 1],
                self.tsta[0 + self.seq_len],
                self.tsta[0 + self.seq_len + self.horizon - 1],
            ],
            'sample_last': [
                self.tsta[self.n_sample - 1],
                self.tsta[self.n_sample - 1 + self.seq_len - 1],
                self.tsta[self.n_sample - 1 + self.seq_len],
                self.tsta[self.n_sample - 1 + self.seq_len + self.horizon - 1],
            ],
            'means_for_scale': self.means_for_scale,
            'stds_for_scale': self.stds_for_scale,
        }

    def set_means_stds_for_scale(self, means, stds):
        self.means_for_scale = means
        self.stds_for_scale = stds
        self.means_for_scale_tensor = TSDataset.to_tensor(self.means_for_scale)
        self.stds_for_scale_tensor = TSDataset.to_tensor(self.stds_for_scale)

    def rescale(self, x):
        return self.means_for_scale_tensor \
            + torch.einsum('k,ijk->ijk', (self.stds_for_scale_tensor, x))

    def __len__(self):
        return self.n_sample

    def __getitem__(self, idx):
        # idx :  Start of the reference window
        # idx + seq_len - 1 :  End of the reference window (current time) (= idx_1 - 1)
        # idx + seq_len - 1 + 1 :  Start of the prediction window (= idx_1)
        # idx + seq_len - 1 + horizon :  End of the prediction window
        idx_1 = idx + self.seq_len
        idx_2 = idx + self.seq_len + self.horizon

        tsta = self.tsta[idx:idx_1]
        tste = self.tste[idx:idx_1]
        data = self.df.iloc[idx:idx_1, :].values
        datass = (self.df.iloc[idx:idx_1, :].values - self.means_for_scale) / self.stds_for_scale

        tsta_future = self.tsta[idx_1:idx_2]
        tste_future = self.tste[idx_1:idx_2]
        data_future = self.df.iloc[idx_1:idx_2, :].values
        datass_future = (self.df.iloc[idx_1:idx_2, :].values - self.means_for_scale) / self.stds_for_scale

        return TSDataset.TSBatch(tsta, tste, data, datass, tsta_future, tste_future, data_future, datass_future)

    @staticmethod
    def collate_fn(batch):
        tsta = np.array([v[0] for v in batch])  # batch_size, seq_len
        tste = TSDataset.to_tensor(np.array([v[1] for v in batch]))  # batch_size, seq_len
        data = TSDataset.to_tensor(np.array([v[2] for v in batch]))  # batch_size, seq_len, num_of_roads
        datass = TSDataset.to_tensor(np.array([v[3] for v in batch]))  # batch_size, seq_len, num_of_roads
        tsta_future = np.array([v[4] for v in batch])  # batch_size, pred_len
        tste_future = TSDataset.to_tensor(np.array([v[5] for v in batch]))  # batch_size, pred_len
        data_future = TSDataset.to_tensor(np.array([v[6] for v in batch]))  # batch_size, pred_len, num_of_roads
        datass_future = TSDataset.to_tensor(np.array([v[7] for v in batch]))  # batch_size, pred_len, num_of_roads
        return TSDataset.TSBatch(tsta, tste, data, datass, tsta_future, tste_future, data_future, datass_future)

    @staticmethod
    def debug_print_batch(batch):
        print(f' * {batch.tsta.shape=}, {batch.data.size()=}')
        print(f'   * {batch.tsta[0][ 0]=}')
        print(f'   * {batch.tsta[0][-1]=}')
        print(f'   * {batch.tsta[batch.tsta.shape[0] - 1][ 0]=}')
        print(f'   * {batch.tsta[batch.tsta.shape[0] - 1][-1]=}')
        print(f' * {batch.tsta_future.shape=}, {batch.data_future.size()=}')


class TSDataManager:
    def __init__(
        self,
        path,
        colname_timestamp,
        white_list,
        seq_len,
        pred_len,
        step_start=0,
        step_width=1,
    ):
        self.path = path
        self.colname_timestamp = colname_timestamp
        if isinstance(white_list, str) and (white_list != ''):
            self.white_list = white_list.split(',')
        elif isinstance(white_list, list):
            self.white_list = [str(w) for w in white_list]
        else:
            self.white_list = ''
        self.step_start = step_start
        self.step_width = step_width
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.load_data()

    def load_data(self):
        df = pd.read_csv(self.path)
        df.columns = [str(col) for col in df.columns]
        cols = []
        if self.white_list != '':
            cols += self.white_list
        else:
            cols += [col for col in df.columns if col != self.colname_timestamp]
        self.cols_org = cols
        df = df.loc[:, [self.colname_timestamp] + cols]
        self.n_channel = len(df.columns) - 1
        df.columns = ['timestamp'] + [f'y{i}' for i in range(self.n_channel)]
        n_rows = len(df)
        df.insert(1, 'timestep', [self.step_start + i * self.step_width for i in range(n_rows)])
        self.df = df

    def get_range(self, data_range):
        n_rows = len(self.df)
        n_front = n_rows - (self.seq_len - 1) - self.pred_len
        i_start = int(n_front * data_range[0])
        i_end = int(n_front * data_range[1]) + (self.seq_len - 1) + self.pred_len
        return self.df.iloc[i_start:i_end, :]

    def get_dataset(self, logger, data_range):
        df = self.get_range(data_range)
        return TSDataset(logger, df, self.seq_len, self.pred_len, self.cols_org)

    def get_data_loader(
        self,
        logger,
        data_range,
        data_range_for_scale,
        batch_sampler,
        batch_sampler_kwargs,
    ):
        if len(data_range) != 2:
            return None

        dataset = self.get_dataset(logger, data_range)
        dataset_for_scale = self.get_dataset(None, data_range_for_scale)
        means, stds = dataset_for_scale.means, dataset_for_scale.stds
        dataset.set_means_stds_for_scale(means, stds)
        gc.collect()

        kwargs = batch_sampler.filter_kwargs(batch_sampler_kwargs)
        batch_sampler = batch_sampler(dataset.n_sample, **kwargs)

        return DataLoader(
            dataset, batch_sampler=batch_sampler,
            collate_fn=TSDataset.collate_fn)
