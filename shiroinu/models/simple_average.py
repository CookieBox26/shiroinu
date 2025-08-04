from shiroinu.models.base_model import BaseModel
import torch


class BaseSimpleAverage(BaseModel):
    def __init__(self, seq_len, pred_len, period_len):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period_len = period_len
        assert seq_len % self.period_len == 0

    def extract_input(self, batch):
        return batch.data[:, -self.seq_len:, :]

    def extract_target(self, batch):
        return batch.data_future[:, :self.pred_len]

    def predict(self, batch):
        input = self.extract_input(batch)
        output = self(input)
        return output


class SimpleAverage(BaseSimpleAverage):
    def __init__(self, seq_len, pred_len, period_len, decay_rate=1.0, max_n_period=10):
        super().__init__(seq_len, pred_len, period_len)
        self.decay_rate = decay_rate
        self.w_base = torch.tensor(
            [self.decay_rate**j for j in reversed(range(max_n_period))],
            dtype=torch.float, device=self.device,
        )

    def forward(self, x):
        batch_size, _, _ = x.shape
        #    batch_size, seq_len, n_channel
        # -> batch_size, n_period, period_len, n_channel
        x_view = x.view(batch_size, -1, self.period_len, n_channel)
        n_period = x_view.shape[1]
        w = self.w_base[(- n_period):]
        w = w / w.sum()
        return torch.einsum('j,ijkl->ikl', (w, x_view))


class SimpleAverageTrainable(BaseSimpleAverage):
    def __init__(self, seq_len, pred_len, period_len, n_channel):
        super().__init__(seq_len, pred_len, period_len)
        self.n_channel = n_channel
        self.decay_rate = torch.nn.Parameter(torch.full((n_channel,), 0.7))

    def forward(self, x):
        batch_size, _, _ = x.shape
        #    batch_size, seq_len, n_channel
        # -> batch_size, n_period, period_len, n_channel
        x_view = x.view(batch_size, -1, self.period_len, self.n_channel)
        n_period = x_view.shape[1]
        w = torch.tensor([  # n_period, n_channel
            [decay_rate_**j for decay_rate_ in self.decay_rate]
            for j in reversed(range(n_period))
        ], dtype=torch.float, device=self.device, requires_grad=True)
        w = w / w.sum(dim=0)
        return torch.einsum('jl,ijkl->ikl', (w, x_view))
