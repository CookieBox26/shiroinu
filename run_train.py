from shiroinu.data_manager import TSDataManager
from shiroinu.batch_sampler import BatchSampler
from nbeats_pytorch.model import NBeatsNet
import numpy as np
import torch
from shiroinu import get_conf_and_logger, fix_seed, load_class, load_instance
import sys


def predict_and_backward(batch, criterion, model, optimizer):
    optimizer.zero_grad()
    pred, info = model(*model.extract_args(batch))
    true = model.extract_true(batch)
    loss = model.get_loss(pred, info, true, criterion, backward=True)
    optimizer.step()
    return pred, true, loss, info


def predict_only(batch, criterion, model):
    with torch.no_grad():
        pred, info = model(*model.extract_args(batch))
        true = model.extract_true(batch)
        loss = model.get_loss(pred, info, true, criterion)
    return pred, true, loss, info


def run_task_train(data_loader, criterion, model, optimizer):
    loss_total = 0.0
    for i_batch, batch in enumerate(data_loader):
        pred, true, loss, info = predict_and_backward(batch, criterion, model, optimizer)
        loss_total += batch.tsta_future.shape[0] * loss.item()
    return loss_total / data_loader.dataset.n_sample


def run_task_valid(data_loader, criterion, model):
    loss_total = 0.0
    for i_batch, batch in enumerate(data_loader):
        pred, true, loss, info = predict_only(batch, criterion, model)
        loss_total += batch.tsta_future.shape[0] * loss.item()
    return loss_total / data_loader.dataset.n_sample


def run_task(logger, dm, criteria, model, task, batch_size_valid):
    logger.start_task()

    cls_optimizer = load_class(task.optimizer.path)
    optimizer = cls_optimizer(model.parameters(), **task.optimizer.params)
    lr_scheduler = None
    if task.lr_scheduler.path != '':
        cls_lr_scheduler = load_class(task.lr_scheduler.path)
        lr_scheduler = cls_lr_scheduler(optimizer, **task.lr_scheduler.params)

    data_loader_train = dm.get_data_loader(
        logger=logger, data_range=task.train_range,
        data_range_for_scale=task.train_range,
        batch_sampler=load_class(task.batch_sampler.path),
        batch_sampler_kwargs=task.batch_sampler.params,
    )
    data_loader_valid = dm.get_data_loader(
        logger=logger, data_range=task.valid_range,
        data_range_for_scale=task.train_range,
        batch_sampler=BatchSampler,
        batch_sampler_kwargs={'batch_size': batch_size_valid},
    )
    logger.add_info('means_for_scale', data_loader_valid.dataset.means_for_scale)
    logger.add_info('stds_for_scale', data_loader_valid.dataset.stds_for_scale)

    loss_valid_best = float('inf')
    early_stop_counter = 0
    stop = False
    for i_epoch in range(task.n_epoch):
        logger.start_epoch()
        logger.add_info_epoch('learning_rate', optimizer.param_groups[0]['lr'])
        loss_train = run_task_train(data_loader_train, criteria[0], model, optimizer)
        logger.add_info_epoch('loss_0_per_sample_train', loss_train)
        if lr_scheduler is not None:
            lr_scheduler.step()
        for i_criterion, criterion in enumerate(criteria):
            loss_valid = run_task_valid(data_loader_valid, criterion, model)
            logger.add_info_epoch(f'loss_{i_criterion}_per_sample_valid', loss_valid)
            if i_criterion == 0:
                if loss_valid < loss_valid_best:
                    logger.save_model(model, '_best')
                    loss_valid_best = loss_valid
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                if (task.early_stop) and (early_stop_counter >= 3):
                    stop = True
        if stop:
            break

    logger.save_model(model, '_last')
    logger.end_task()


def main():
    conf, logger = get_conf_and_logger(sys.argv[1], 'train')
    dm = TSDataManager(**conf.data)
    criteria = []
    for criterion in conf.criteria:
        criteria.append(load_instance(**criterion))

    model = None
    for task in conf.tasks:
        if (model is None) or task.reset_model:
            fix_seed()
            model = load_instance(**conf.model)
        run_task(logger, dm, criteria, model, task, conf.batch_size_valid)


if __name__ == '__main__':
    main()
