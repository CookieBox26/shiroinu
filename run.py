from shiroinu.data_manager import TSDataManager
from shiroinu.batch_sampler import BatchSampler
from shiroinu import get_conf_and_logger, fix_seed, load_class, load_instance, create_instance
from shiroinu.report import ReportWriter
import torch
import argparse


def run_task_train(data_loader, criterion, model, optimizer):
    loss_total = 0.0
    for i_batch, batch in enumerate(data_loader):
        optimizer.zero_grad()
        loss_, _, _, _ = model.get_loss(batch, criterion)
        loss = loss_ if isinstance(loss_, torch.Tensor) else loss_[0]
        loss.backward()
        optimizer.step()
        loss_total += batch.tsta_future.shape[0] * loss.item()
    return loss_total / data_loader.dataset.n_sample


def run_task_valid(data_loader, criterion, model):
    loss_total = 0.0
    with torch.no_grad():
        for i_batch, batch in enumerate(data_loader):
            loss_, _, _, _ = model.get_loss(batch, criterion)
            loss = loss_ if isinstance(loss_, torch.Tensor) else loss_[0]
            loss_total += batch.tsta_future.shape[0] * loss.item()
    return loss_total / data_loader.dataset.n_sample


def run_task(conf, logger, dm, model, task, batch_size_eval):
    """Training-type task
    """
    data_loader_train = dm.get_data_loader(
        logger=logger,
        data_range=task.train_range,
        batch_sampler=load_class(task.batch_sampler.path),
        batch_sampler_kwargs=task.batch_sampler.params,
    )
    logger.add_info('data_train', data_loader_train.dataset.get_info_for_logger())

    data_loader_valid = None
    if task.valid_range is not None:
        data_loader_valid = dm.get_data_loader(
            logger=logger,
            data_range=task.valid_range,
            batch_sampler=BatchSampler,
            batch_sampler_kwargs={'batch_size': batch_size_eval},
        )
        logger.add_info('data_valid', data_loader_valid.dataset.get_info_for_logger())

    if (model is None) or task.reset_model:
        fix_seed()
        model_settings = conf.get_model(**task.model)
        model = create_instance(dataset=data_loader_train.dataset, **model_settings)
        print(
            f'{model.__class__.__name__}({model.count_trainable_parameters()}) '
            f'loaded to {model.device}'
        )

    criterion_target = load_instance(**task.criterion_target)
    cls_optimizer = load_class(task.optimizer.path)
    optimizer = cls_optimizer(model.parameters(), **task.optimizer.params)
    lr_scheduler = None
    if task.lr_scheduler.path != '':
        cls_lr_scheduler = load_class(task.lr_scheduler.path)
        lr_scheduler = cls_lr_scheduler(optimizer, **task.lr_scheduler.params)

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
        logger.add_info_epoch('loss_per_sample_train', loss_train)
        if lr_scheduler is not None:
            lr_scheduler.step()
        if data_loader_valid is None:
            continue

        loss_valid = run_task_valid(data_loader_valid, criterion_target, model)
        logger.add_info_epoch(f'loss_per_sample_valid', loss_valid)
        if loss_valid < loss_valid_best:
            logger.save_model(model, suffix='_best')
            loss_valid_best = loss_valid
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if (task.early_stop) and (early_stop_counter >= 5):
            stop = True
        if stop:
            break

    logger.save_model(model, suffix='_end')
    return model


def run_task_eval(conf, logger, dm, task, batch_size_eval):
    """Evaluation-type task
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = dm.get_data_loader(
        logger=logger,
        data_range=task.valid_range,
        batch_sampler=BatchSampler,
        batch_sampler_kwargs={'batch_size': batch_size_eval},
    )
    logger.add_info('data', data_loader.dataset.get_info_for_logger())

    models = []
    for model_ in task.models:
        model_settings = conf.get_model(**model_)
        models.append(create_instance(dataset=data_loader.dataset, **model_settings))
    criterion = load_instance(**task.criterion_eval)

    n_model = len(models)
    pred_len = models[0].pred_len
    for i_model in range(n_model):
        assert models[i_model].pred_len == pred_len, 'Output length mismatch.'

    data_loss_detail = [torch.empty(0, dm.n_channel, device=device)] * n_model
    with torch.no_grad():
        for i_batch, batch in enumerate(data_loader):
            true = batch.data_future[:, :pred_len]
            if i_batch == 0:
                logger.save_array('sample_0_true', true[0])
                logger.save_array('sample_0_tsta', batch.tsta_future[0])
            for i_model, model in enumerate(models):
                pred = model.predict(batch)
                _, _, loss_detail = criterion(pred, true)
                loss_detail = loss_detail.detach().clone()
                data_loss_detail[i_model] = torch.cat([data_loss_detail[i_model], loss_detail], dim=0)
                if i_batch == 0:
                    logger.save_array(f'sample_0_model_{i_model}', pred[0])

    # print(data_loss_detail[i_model].size())  # n_sample, n_channel
    for i_model in range(n_model):
        print(f'model_{i_model}:', data_loss_detail[i_model].mean().item())
    loss_per_sample = [data_loss_detail[i_model].mean(dim=0).tolist() for i_model in range(n_model)]
    logger.add_info('loss_per_sample', loss_per_sample)
    percentiles = [
        torch.quantile(
            data_loss_detail[i_model],
            torch.tensor(task.percentile_points, device=device),
            dim=0,
        ).tolist()
        for i_model in range(n_model)
    ]
    logger.add_info('percentiles', percentiles)


def run_tasks(conf, logger, li_skip_task_id):
    dm = TSDataManager(**conf.data)
    model = None
    for i_task, task in enumerate(conf.tasks):
        if i_task in li_skip_task_id:
            logger.skip_task()
            continue

        logger.start_task()
        if task.task_type == 'train':
            model = run_task(conf, logger, dm, model, task, conf.batch_size_eval)
        if task.task_type == 'eval':
            run_task_eval(conf, logger, dm, task, conf.batch_size_eval)
        logger.end_task()


def run(
    conf_file,
    skip_task_ids,
    report_only,
    clear_logs,
    quiet,
    image_format,
    separate_image,
    dpi,
    max_n_graph,
):
    conf, logger = get_conf_and_logger(conf_file, clear_logs)
    logger.print_epoch = (not quiet)
    li_skip_task_id = [int(i) for i in skip_task_ids.split(',') if i != '']
    if not report_only:
        run_tasks(conf, logger, li_skip_task_id)
    ReportWriter.create(
        conf_file,
        image_format=image_format,
        embed_image=(not separate_image),
        dpi=dpi,
        max_n_graph=max_n_graph,
    )()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf_file')
    parser.add_argument('-s', '--skip_task_ids', type=str, default='')
    parser.add_argument('-r', '--report_only', action='store_true')
    parser.add_argument('-c', '--clear_logs', action='store_true')
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('-f', '--image_format', type=str, default='svg')
    parser.add_argument('-i', '--separate_image', action='store_true')
    parser.add_argument('--dpi', type=int, default=100)
    parser.add_argument('--max_n_graph', type=int, default=1000)
    args = parser.parse_args()
    run(
        args.conf_file,
        args.skip_task_ids,
        args.report_only,
        args.clear_logs,
        args.quiet,
        args.image_format,
        args.separate_image,
        args.dpi,
        args.max_n_graph,
    )
