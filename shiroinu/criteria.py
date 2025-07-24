import torch


class MAELoss(torch.nn.Module):
    def __init__(self, n_channel):
        super(MAELoss, self).__init__()
        self.w_channel = torch.ones(n_channel, dtype=torch.float)
        self.w_channel /= self.w_channel.sum()
    def forward(self, pred, true):
        ae = torch.abs(pred - true)  # batch_size, pred_len, num_of_roads
        mae = ae.mean(dim=1)  # batch_size, num_of_roads
        return torch.einsum('j,ij->ij', (self.w_channel, mae)), mae.detach().clone()


class ExceedanceRate(torch.nn.Module):
    def __init__(self, n_channel, threshold=0.01):
        super(ExceedanceRate, self).__init__()
        self.w_channel = torch.ones(n_channel, dtype=torch.float)
        self.w_channel /= self.w_channel.sum()
        self.threshold = threshold
    def forward(self, pred, true):
        ae = torch.abs(pred - true)  # batch_size, pred_len, num_of_roads
        binary = (ae >= self.threshold).int().float()
        er = binary.mean(dim=1)  # batch_size, num_of_roads
        return torch.einsum('j,ij->ij', (self.w_channel, er)), er.detach().clone()


class DiffLoss(torch.nn.Module):
    def __init__(self, n_channel, threshold=0.005):
        super(DiffLoss, self).__init__()
        self.w_channel = torch.ones(n_channel, dtype=torch.float)
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
