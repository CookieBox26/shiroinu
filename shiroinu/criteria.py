import torch
from abc import ABC


class BaseLoss(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MSELoss(BaseLoss):
    def set_w_channel(self, n_channel):
        self.w_channel = torch.ones(n_channel, dtype=torch.float, device=self.device)
        self.w_channel /= self.w_channel.sum()
    def __init__(self, n_channel=0):
        super().__init__()
        self.w_channel = None
        if n_channel > 0:
            self.set_w_channel(n_channel)
    def calc_loss(self, pred, true):
        return (pred - true) ** 2
    def forward(self, pred, true):
        if self.w_channel is None:
            self.set_w_channel(pred.size()[2])
        loss = self.calc_loss(pred, true)  # batch_size, pred_len, num_of_roads
        me_of_each_sample_channel = loss.mean(dim=1)  # batch_size, num_of_roads
        me_of_each_sample = torch.einsum('j,ij->ij', (self.w_channel, me_of_each_sample_channel))
        return (
            me_of_each_sample.mean(),  # (scalar)
            me_of_each_sample,  # batch_size
            me_of_each_sample_channel,  # batch_size, n_channel
        )


class MAELoss(MSELoss):
    def __init__(self, n_channel=0):
        super().__init__(n_channel)
    def calc_loss(self, pred, true):
        return torch.abs(pred - true)


class ExceedanceRate(BaseLoss):
    def __init__(self, n_channel, threshold=0.01):
        super().__init__()
        self.w_channel = torch.ones(n_channel, dtype=torch.float, device=self.device)
        self.w_channel /= self.w_channel.sum()
        self.threshold = threshold
    def forward(self, pred, true):
        ae = torch.abs(pred - true)  # batch_size, pred_len, num_of_roads
        binary = (ae >= self.threshold).int().float()
        er = binary.mean(dim=1)  # batch_size, num_of_roads
        return torch.einsum('j,ij->ij', (self.w_channel, er)), er.detach().clone()


class DiffLoss(BaseLoss):
    def __init__(self, n_channel, threshold=0.005):
        super().__init__()
        self.w_channel = torch.ones(n_channel, dtype=torch.float, device=self.device)
        self.w_channel /= self.w_channel.sum()
        self.threshold = threshold
    def forward(self, pred, true):
        pred_diff = pred[:, 1:, :] - pred[:, :-1, :]  # batch_size, pred_len - 1, num_of_roads
        true_diff = true[:, 1:, :] - true[:, :-1, :]  # batch_size, pred_len - 1, num_of_roads
        pred_diff = torch.where(pred_diff <= -self.threshold, -1, torch.where(pred_diff >= self.threshold, 1, 0))
        true_diff = torch.where(true_diff <= -self.threshold, -1, torch.where(true_diff >= self.threshold, 1, 0))
        diff = pred_diff * true_diff
        diff = torch.where(diff == -1, 1, 0).float()
        dl = diff.mean(dim=1)  # batch_size, num_of_roads
        return torch.einsum('j,ij->ij', (self.w_channel, dl)), dl.detach().clone()
