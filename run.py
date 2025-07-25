from shiroinu.data_manager import TSDataManager
from shiroinu.batch_sampler import BatchSampler
from shiroinu import get_conf_and_logger, fix_seed, load_class, load_instance
import torch
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


def run_task(logger, dm, criterion_target, criteria, model, task, batch_size_eval):
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
        batch_sampler_kwargs={'batch_size': batch_size_eval},
    )
    logger.add_info('means_for_scale', data_loader_valid.dataset.means_for_scale)
    logger.add_info('stds_for_scale', data_loader_valid.dataset.stds_for_scale)

    loss_valid_best = float('inf')
    early_stop_counter = 0
    stop = False
    n_epoch_ = task.n_epoch
    if task.n_epoch_ref > -1:
        print(f'Reuse the best epoch count from task {task.n_epoch_ref}')
        n_epoch_ = logger.d_epoch_id_best[task.n_epoch_ref] + 1
    for i_epoch in range(n_epoch_):
        logger.start_epoch()
        logger.add_info_epoch('learning_rate', optimizer.param_groups[0]['lr'])
        loss_train = run_task_train(data_loader_train, criterion_target, model, optimizer)
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


def run_task_eval(logger, dm, criterion, models, task, batch_size_eval):
    logger.start_task()
    data_loader = dm.get_data_loader(
        logger=logger, data_range=task.valid_range,
        data_range_for_scale=task.train_range,
        batch_sampler=BatchSampler,
        batch_sampler_kwargs={'batch_size': batch_size_eval},
    )

    model_0 = models[0]
    model_1 = models[1]

    data_loss0_detail_0 = torch.empty(0, dm.n_channel)
    data_loss0_detail_1 = torch.empty(0, dm.n_channel)

    for i_batch, batch in enumerate(data_loader):
        true = model_0.extract_true(batch)
        with torch.no_grad():
            pred_0, _ = model_0(*model_0.extract_args(batch))
            pred_1, _ = model_1(*model_1.extract_args(batch))
            pred_0 = model_0.rescale(data_loader.dataset, pred_0)
            pred_1 = model_1.rescale(data_loader.dataset, pred_1)
            loss0_0, loss0_detail_0 = criterion(pred_0, true)
            loss0_1, loss0_detail_1 = criterion(pred_1, true)

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
    logger.end_task()


def main():
    conf, logger = get_conf_and_logger(sys.argv[1])
    dm = TSDataManager(**conf.data)

    criteria = []
    for criterion in conf.criteria:
        criteria.append(load_instance(**criterion))

    model = None
    for task in conf.tasks:
        if task.task_type == 'train':
            criterion_target = load_instance(**task.criterion_target)
            if (model is None) or task.reset_model:
                fix_seed()
                model_settings = conf.get_model(**task.model)
                model = load_instance(**model_settings)
            run_task(logger, dm, criterion_target, criteria, model, task, conf.batch_size_eval)
        if task.task_type == 'eval':
            criterion_eval = load_instance(**task.criterion_eval)
            models_eval = []
            for model_ in task.models:
                model_settings = conf.get_model(**model_)
                models_eval.append(load_instance(**model_settings))
            run_task_eval(logger, dm, criterion_eval, models_eval, task, conf.batch_size_eval)


if __name__ == '__main__':
    main()
