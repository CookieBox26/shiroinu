from shiroinu.scaler import StandardScaler
from shiroinu import load_instance
import torch


def test_standard_scaler():
    scaler = StandardScaler(
        means_=[0, 2.5, 5],
        stds_=[1, 2.5, 5],
    )
    batch_0 = torch.tensor([[
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
    ]])
    batch_1 = scaler.scale(batch_0)
    expected = torch.tensor([10., 3., 1.])
    assert torch.allclose(batch_1[0][0], expected)
    batch_1 = scaler.rescale(batch_1)
    assert torch.allclose(batch_1, batch_0)


def test_mse_loss():
    conf_crit = {'path': 'shiroinu.criteria.MSELoss', 'params': {}}
    criterion = load_instance(**conf_crit)
