from shiroinu.data_manager import TSDataManager
from shiroinu.batch_sampler import BatchSampler
from shiroinu import get_conf_and_logger, load_instance
import torch
torch.set_printoptions(precision=5)
import sys


def main():
    conf, logger = get_conf_and_logger(sys.argv[1], 'eval')

    dm = TSDataManager(**conf.data)
    data_loader = dm.get_data_loader(
        logger=logger,
        data_range=conf.tasks_eval[0].valid_range,
        batch_sampler=BatchSampler,
        batch_sampler_kwargs={'batch_size': conf.batch_size_valid},
    )
    data_for_scale = dm.get_dataset(None, conf.tasks_eval[0].train_range)
    means, stds = data_for_scale.means, data_for_scale.stds
    data_loader.dataset.set_means_stds_for_scale(means, stds)

    models = []
    for conf_model in conf.models_eval:
        models.append(load_instance(**conf_model))
    model_0 = models[0]
    model_1 = models[1]

    criterion0 = load_instance(**conf.criterion_eval)

    data_loss0_detail_0 = torch.empty(0, dm.n_channel)
    data_loss0_detail_1 = torch.empty(0, dm.n_channel)

    for i_batch, batch in enumerate(data_loader):
        true = model_0.extract_true(batch)
        with torch.no_grad():
            pred_0, _ = model_0(*model_0.extract_args(batch))
            pred_1, _ = model_1(*model_1.extract_args(batch))
            pred_0 = model_0.rescale(data_loader.dataset, pred_0)
            pred_1 = model_1.rescale(data_loader.dataset, pred_1)
            loss0_0, loss0_detail_0 = criterion0(pred_0, true)
            loss0_1, loss0_detail_1 = criterion0(pred_1, true)

        data_loss0_detail_0 = torch.cat([data_loss0_detail_0, loss0_detail_0], dim=0)
        data_loss0_detail_1 = torch.cat([data_loss0_detail_1, loss0_detail_1], dim=0)

    logger.log(data_loss0_detail_0.size())  # n_sample, num_of_roads
    total_loss0_0 = data_loss0_detail_0.mean(dim=0)
    total_loss0_1 = data_loss0_detail_1.mean(dim=0)
    logger.log(total_loss0_0)
    logger.log(total_loss0_1)

    percentiles = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    logger.log(torch.quantile(data_loss0_detail_0, percentiles, dim=0))
    logger.log(torch.quantile(data_loss0_detail_1, percentiles, dim=0))
    logger.close()


if __name__ == '__main__':
    main()
