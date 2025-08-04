from shiroinu.models.base_model import BaseModel
from shiroinu.models.simple_average import SimpleAverage
from shiroinu.scaler import StandardScaler, IqrScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def moving_avg(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class BaseDLinear(BaseModel):
    def __init__(self, seq_len, pred_len, kernel_size, bias):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decompsition = series_decomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len, bias=bias)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len, bias=bias)

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
        return x, {'seasonal': seasonal_output, 'trend': trend_output}

    def extract_input(self, batch):
        return self.scaler.scale(batch.data[:, -self.seq_len:, :])

    def extract_target(self, batch):
        return self.scaler.scale(batch.data_future[:, :self.pred_len])

    def predict(self, batch):
        input = self.extract_input(batch)
        output, _ = self(input)
        return self.scaler.rescale(output)


class DLinear(BaseDLinear):
    data_based_hyperparams = ['means_', 'stds_']
    def __init__(self, seq_len, pred_len, kernel_size, bias, means_, stds_):
        super().__init__(seq_len, pred_len, kernel_size, bias)
        self.scaler = StandardScaler(means_, stds_)


class DLinearIqr(BaseDLinear):
    data_based_hyperparams = ['q1s_', 'q2s_', 'q3s_']
    def __init__(self, seq_len, pred_len, kernel_size, bias, q1s_, q2s_, q3s_):
        super().__init__(seq_len, pred_len, kernel_size, bias)
        self.scaler = IqrScaler(q1s_, q2s_, q3s_)


class BaseDLinearRes(BaseDLinear):
    def __init__(self, seq_len, pred_len, kernel_size, bias, period_len, decay_rate=1.0):
        super().__init__(seq_len, pred_len, kernel_size, bias)
        self.baseline = SimpleAverage(
            seq_len=seq_len, pred_len=pred_len, period_len=period_len,
            decay_rate=decay_rate,
        )

    def extract_input(self, batch):
        input_ = self.scaler.scale(batch.data[:, -self.seq_len:, :])
        naive = self.baseline(input_)
        return input_ - naive.repeat(1, int(self.seq_len / self.baseline.period_len), 1)

    def extract_target(self, batch):
        naive = self.baseline(self.scaler.scale(batch.data[:, -self.seq_len:, :]))
        target = self.scaler.scale(batch.data_future[:, :self.pred_len])
        return target - naive

    def predict(self, batch):
        naive = self.baseline(self.scaler.scale(batch.data[:, -self.seq_len:, :]))
        input = self.extract_input(batch)
        output, _ = self(input)
        return self.scaler.rescale(naive + output)


class DLinearRes(BaseDLinearRes):
    data_based_hyperparams = ['means_', 'stds_']
    def __init__(
        self, seq_len, pred_len, kernel_size, bias, period_len,
        means_, stds_, decay_rate=1.0,
    ):
        super().__init__(seq_len, pred_len, kernel_size, bias, period_len, decay_rate)
        self.scaler = StandardScaler(means_, stds_)


class DLinearResIqr(BaseDLinearRes):
    data_based_hyperparams = ['q1s_', 'q2s_', 'q3s_']
    def __init__(
        self, seq_len, pred_len, kernel_size, bias, period_len,
        q1s_, q2s_, q3s_, decay_rate=1.0,
    ):
        super().__init__(seq_len, pred_len, kernel_size, bias, period_len, decay_rate)
        self.scaler = IqrScaler(q1s_, q2s_, q3s_)


class DLinearSparse(DLinear):
    def __init__(
        self,
        seq_len,
        pred_len,
        kernel_size,
        bias=False,
    ):
        super().__init__(seq_len, pred_len, kernel_size)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decompsition = series_decomp(kernel_size)
        self.not_frozen = [i for i in range(0, seq_len, pred_len)]
        self.Linear_Seasonal = nn.Linear(len(self.not_frozen), self.pred_len, bias=bias)
        self.Linear_Trend = nn.Linear(len(self.not_frozen), self.pred_len, bias=bias)

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        Linear_Seasonal = torch.zeros([self.pred_len, self.seq_len], dtype=torch.float).to(self.device)
        Linear_Trend = torch.zeros([self.pred_len, self.seq_len], dtype=torch.float).to(self.device)
        for i, i_nf in enumerate(self.not_frozen):
            for j in range(self.pred_len):
                Linear_Seasonal[j, i_nf + j] = self.Linear_Seasonal.weight[j, i]
                Linear_Trend[j, i_nf + j] = self.Linear_Trend.weight[j, i]
        seasonal_output = torch.einsum('ijk,lk->ijl', (seasonal_init, Linear_Seasonal))
        trend_output = torch.einsum('ijk,lk->ijl', (trend_init, Linear_Trend))
        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
        return x, {'seasonal': seasonal_output, 'trend': trend_output}


class DLinears(DLinear):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(
        self,
        seq_len,
        pred_len,
        kernel_size,
        n_channel,
        state_paths=None,
        bias=False,
    ):
        super().__init__(seq_len, pred_len, kernel_size)
        self.n_channel = n_channel
        self.dlinears = nn.ModuleList()
        if state_paths is None:
            state_paths = [None] * self.n_channel
        for i in range(self.n_channel):
            self.dlinears.append(DLinear(seq_len, pred_len, kernel_size, state_paths[i], bias))

    def forward(self, x):
        output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
        for i in range(self.n_channel):
            pred, _ = self.dlinears[i](x[:, :, i:(i+1)])
            output[:, :, i] = pred.squeeze(-1)
        return output, {}


class DLinearSparses(DLinear):
    def __init__(
        self,
        seq_len,
        pred_len,
        kernel_size,
        n_channel,
        state_paths=None,
    ):
        super().__init__(seq_len, pred_len, kernel_size)
        self.n_channel = n_channel
        self.dlinears = nn.ModuleList()
        if state_paths is None:
            state_paths = [None] * self.n_channel
        for i in range(self.n_channel):
            self.dlinears.append(DLinearSparse(seq_len, pred_len, kernel_size, state_paths[i]))

    def forward(self, x):
        output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
        for i in range(self.n_channel):
            pred, _ = self.dlinears[i](x[:, :, i:(i+1)])
            output[:, :, i] = pred.squeeze(-1)
        return output, {}
