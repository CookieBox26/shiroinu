import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
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


class DLinear(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(
        self,
        seq_len,
        pred_len,
        kernel_size,
        state_path=None,
        bias=False,
    ):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decompsition = series_decomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len, bias=bias)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len, bias=bias)
        if (state_path is not None) and (state_path != ''):
            self.load_state_dict(torch.load(state_path))
            self.eval()
        self.to(DLinear.device)

    def extract_args(self, batch):
        return [batch.datass[:, -self.seq_len:, :]]

    def extract_true(self, batch):
        return batch.datass_future[:, :self.pred_len]

    def get_loss(self, pred, info, true, criterion, backward=False):
        loss = criterion(pred, true)
        if backward:
            loss.backward()
        return loss

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
        return x, {'seasonal': seasonal_output, 'trend': trend_output}

    def rescale(self, dataset, x):
        means = dataset.to_tensor(dataset.means_for_scale)
        stds = dataset.to_tensor(dataset.stds_for_scale)
        return means + torch.einsum('k,ijk->ijk', (stds, x))


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
        super(DLinears, self).__init__(seq_len, pred_len, kernel_size)
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


class DLinearSparse(DLinear):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(
        self,
        seq_len,
        pred_len,
        kernel_size,
        state_path=None,
        bias=False,
    ):
        super(DLinearSparse, self).__init__(seq_len, pred_len, kernel_size)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decompsition = series_decomp(kernel_size)

        self.not_frozen = [i for i in range(0, seq_len, pred_len)]
        self.Linear_Seasonal = nn.Linear(len(self.not_frozen), self.pred_len, bias=bias)
        self.Linear_Trend = nn.Linear(len(self.not_frozen), self.pred_len, bias=bias)
        if (state_path is not None) and (state_path != ''):
            self.load_state_dict(torch.load(state_path))
            self.eval()
        self.to(DLinear.device)

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        Linear_Seasonal = torch.zeros([self.pred_len, self.seq_len], dtype=torch.float).to(DLinear.device)
        Linear_Trend = torch.zeros([self.pred_len, self.seq_len], dtype=torch.float).to(DLinear.device)
        for i, i_nf in enumerate(self.not_frozen):
            for j in range(self.pred_len):
                Linear_Seasonal[j, i_nf + j] = self.Linear_Seasonal.weight[j, i]
                Linear_Trend[j, i_nf + j] = self.Linear_Trend.weight[j, i]
        seasonal_output = torch.einsum('ijk,lk->ijl', (seasonal_init, Linear_Seasonal))
        trend_output = torch.einsum('ijk,lk->ijl', (trend_init, Linear_Trend))
        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
        return x, {'seasonal': seasonal_output, 'trend': trend_output}


class DLinearSparses(DLinear):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(
        self,
        seq_len,
        pred_len,
        kernel_size,
        n_channel,
        state_paths=None,
    ):
        super(DLinearSparses, self).__init__(seq_len, pred_len, kernel_size)
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
