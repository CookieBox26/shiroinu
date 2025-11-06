from shiroinu.scaler import StandardScaler
from shiroinu import create_instance
import torch
import pytest


def test_standard_scaler():
    scaler = StandardScaler(  # Scaling mean and std for 3 roads
        means_=[0, 2.5, 5],
        stds_=[1, 2.5, 5],
    )
    batch_0 = torch.tensor([[  # A batch with 3 roads and 4 steps
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
    """Test for Mean Squared Error
    """
    conf_crit = {'path': 'shiroinu.criteria.MSELoss', 'params': {}}
    criterion = create_instance(**conf_crit)
    batch_0 = torch.tensor([[  # A batch with 3 roads and 4 steps
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
    ]])
    batch_1 = torch.tensor([[
        [11., 12., 11.],
        [22., 24., 22.],
        [33., 36., 33.],
        [44., 48., 44.],
    ]])
    error, errors_each_sample, errors_each_roads = criterion(batch_0, batch_1)

    # ( 1^2 + 2^2 + 3^2 + 4^2 ) / 4 = 30 / 4 = 7.5
    # ( 2^2 + 4^2 + 6^2 + 8^2 ) / 4 = 120 / 4 = 30
    expected = torch.tensor([[7.5, 30., 7.5]])
    assert torch.allclose(errors_each_roads, expected)

    # (7.5 + 30 + 7.5) / 3 = 15
    expected = torch.tensor([15.])
    assert torch.allclose(errors_each_sample, expected)
    assert torch.allclose(error, expected[0])


@pytest.mark.parametrize('class_path', [
    'shiroinu.criteria.NMSELoss',
    'shiroinu.criteria.IQRMSELoss',
    'shiroinu.criteria.NMAELoss',
    'shiroinu.criteria.IQRMAELoss',
])
def test_mse_loss_with_scaler(class_path):
    """Test for Mean Squared Error with Scaler
    """
    conf_crit = {'path': class_path, 'params': {}}
    dataset_dummy = type('TSDatasetDummy', (object,), {  # Scaling mean and std for 3 roads
        'means': [0, 2.5, 5],
        'stds': [1, 2.5, 5],
        'q1s': [0 - 0.5, 2.5 - 0.5 * 2.5, 5 - 0.5 * 5],
        'q2s': [0, 2.5, 5],
        'q3s': [0 + 0.5, 2.5 + 0.5 * 2.5, 5 + 0.5 * 5],
    })
    criterion = create_instance(dataset_valid=dataset_dummy, **conf_crit)
    batch_0 = torch.tensor([[  # A batch with 3 roads and 4 steps
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
    ]])
    batch_1 = torch.tensor([[
        [11., 12., 11.],
        [22., 24., 22.],
        [33., 36., 33.],
        [44., 48., 44.],
    ]])
    error, errors_each_sample, errors_each_roads = criterion(batch_0, batch_1)

    if class_path.endswith('MSELoss'):
        # ( 1^2 + 2^2 + 3^2 + 4^2 ) / 4 = 30 / 4 = 7.5
        # ( 2^2 + 4^2 + 6^2 + 8^2 ) / 4 = 120 / 4 = 30
        expected = torch.tensor([[7.5, (30. / 2.5**2), (7.5 / 5**2)]])
        assert torch.allclose(errors_each_roads, expected)

        expected = torch.tensor([(7.5 + (30. / 2.5**2) + (7.5 / 5**2)) / 3.])
        assert torch.allclose(errors_each_sample, expected)
        assert torch.allclose(error, expected[0])
    elif class_path.endswith('MAELoss'):
        # ( 1 + 2 + 3 + 4 ) / 4 = 10 / 4 = 2.5
        # ( 2 + 4 + 6 + 8 ) / 4 = 20 / 4 = 5
        expected = torch.tensor([[2.5, (5. / 2.5), (2.5 / 5)]])
        assert torch.allclose(errors_each_roads, expected)

        expected = torch.tensor([(2.5 + (5. / 2.5) + (2.5 / 5)) / 3.])
        assert torch.allclose(errors_each_sample, expected)
        assert torch.allclose(error, expected[0])
    else:
        assert False
