from shiroinu.models.base_model import BaseModel
import torch
import torch.nn as nn


class SimpleAverage(BaseModel):
    def __init__(
        self,
        seq_len,
        pred_len,
        period_len,
        decay_rate=1.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period_len = period_len
        self.decay_rate = decay_rate

    def forward(self, x):
        batch_size, seq_len, num_of_roads = x.shape
        assert seq_len % self.period_len == 0

        #    batch_size, seq_len, num_of_roads
        # -> batch_size, n_period, period_len, num_of_roads
        x_view = x.view(batch_size, -1, self.period_len, num_of_roads)
        n_period = x_view.shape[1]
        w = torch.tensor([self.decay_rate**i for i in reversed(range(n_period))], dtype=torch.float)
        w = w / w.sum()
        return torch.einsum('j,ijkl->ikl', (w, x_view))

    def extract_input(self, batch):
        return [batch.data[:, -self.seq_len:, :]]

    def extract_target(self, batch):
        return batch.data_future[:, :self.pred_len]

    def predict(self, batch):
        input = self.extract_input(batch)
        output = self(*input)
        return output
