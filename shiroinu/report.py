from shiroinu.config import Config
import shirotsubaki.report
from shirotsubaki.element import Element as Elm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import base64
import io
import toml
import os
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11


def pairs_to_table(pairs):
    tbody = Elm('tbody')
    for pair in pairs:
        tbody.append(Elm('tr'))
        tbody.inner[-1].append(Elm('td', f'<b>{pair[0]}</b>'))
        for v in pair[1]:
            tbody.inner[-1].append(Elm('td', f'{v}'))
    return Elm('table', tbody).set_attr('border', '1')


def _add_picture(output_path, rp, filename, emb=False):
    kwargs = {'format': 'png', 'bbox_inches': 'tight'}
    if emb:
        pic_io_bytes = io.BytesIO()
        plt.savefig(pic_io_bytes, **kwargs)
        pic_io_bytes.seek(0)
        base64_img = base64.b64encode(pic_io_bytes.read()).decode('utf8')
        rp.append(f'<img src="data:image/png;base64, {base64_img}"/>\n')
    else:
        img_dir = os.path.join(output_path, 'img/')
        os.makedirs(img_dir, exist_ok=True)
        plt.savefig(f'{img_dir}{filename}', **kwargs)
        rp.append(f'<img src="img/{filename}"/>\n')


def _plot_loss_graph(info, li_key_loss, ylabel):
    desc = OrderedDict()
    fig, ax = plt.subplots(nrows=1, figsize=(4, 3))
    li_loss = [[epoch[key_loss] for epoch in info['epochs']] for key_loss in li_key_loss]
    x = [i for i in range(len(li_loss[0]))]
    for i, key_loss in enumerate(li_key_loss):
        desc[key_loss] = ax.plot(x, li_loss[i])[0]
    ax.set_xlabel('i_epoch')
    ax.set_ylabel(ylabel)
    ax.grid(axis='both', linestyle='dotted', linewidth=1)
    ax.legend(desc.values(), desc.keys(), loc='upper right')


def _get_channels_df(info, key):
    return pd.DataFrame({
        'cols_org': info[key]['cols_org'],
        'means_for_scale': info[key]['means_for_scale'],
        'stds_for_scale': info[key]['stds_for_scale'],
    }, index=info[key]['cols'])


def _get_ranges_df(conf, info, key):
    n_sample = info[key]['n_sample']
    li = [conf.data['seq_len'], conf.data['pred_len']]
    return pd.DataFrame({
        'sample 0': li + info[key]['sample_first'],
        f'sample {n_sample - 1}': li + info[key]['sample_last'],
    }, index=[
        'seq_len', 'pred_len',
        'Start of Input Window', 'End of Input Window',
        'Start of Prediction Window', 'End of Prediction Window']).T


def _report_task_train(rp, conf, i_task, info):
    task = conf.tasks[i_task]

    for type_ in ['train', 'valid']:
        rp.append(Elm('h3', f'data_{type_}'))
        rp.append(os.path.basename(conf.data['path']))
        rp.append(_get_channels_df(info, f'data_{type_}').to_html())
        rp.append(_get_ranges_df(conf, info, f'data_{type_}').to_html())

    rp.append(Elm('h3', 'Model and Optimizer'))
    model_settings = conf.get_model(**task.model)
    pairs = [
        ('model_path', [model_settings['path']]),
        ('model_params', [model_settings['params']])]
    rp.append(pairs_to_table(pairs))
    pairs = [
        (key, [getattr(task, key).path, getattr(task, key).params])
        for key in ['batch_sampler', 'optimizer', 'lr_scheduler']]
    rp.append(pairs_to_table(pairs))

    if (
        (task.criterion_target['path'] == conf.criteria[0]['path'])
        and (task.criterion_target['params'] == conf.criteria[0]['params'])
    ):
        _plot_loss_graph(
            info, ['loss_0_per_sample_train', 'loss_0_per_sample_valid'],
            task.criterion_target['path'])
        _add_picture(conf.log_dir, rp, f'task_{i_task}_loss_train_valid.png')
    else:
        _plot_loss_graph(info, ['loss_0_per_sample_train'], task.criterion_target['path'])
        _add_picture(conf.log_dir, rp, f'task_{i_task}_loss_train.png')
        _plot_loss_graph(info, ['loss_0_per_sample_valid'], conf.criteria[0]['path'])
        _add_picture(conf.log_dir, rp, f'task_{i_task}_loss_0_valid.png')
    for i_crit in range(1, len(conf.criteria)):
        _plot_loss_graph(info, [f'loss_{i_crit}_per_sample_valid'], conf.criteria[i_crit]['path'])
        _add_picture(conf.log_dir, rp, f'task_{i_task}_loss_{i_crit}_valid.png')

    rp.append(pd.DataFrame({
        'epoch_id_best': [info['epoch_id_best']],
        'loss_valid_best': [info['epochs'][info['epoch_id_best']]['loss_0_per_sample_valid']],
    }).to_html(index=False))


def _report_task_eval(rp, conf, i_task, info):
    task = conf.tasks[i_task]
    n_model = len(task.models)

    rp.append(Elm('h3', 'data'))
    rp.append(os.path.basename(conf.data['path']))
    rp.append(_get_channels_df(info, 'data').to_html())
    rp.append(_get_ranges_df(conf, info, 'data').to_html())

    rp.append(Elm('h3', 'Criterion'))
    pairs = [('criterion_eval', [task.criterion_eval['path'], task.criterion_eval['params']])]
    rp.append(pairs_to_table(pairs))

    rp.append(Elm('h3', 'Models'))
    pairs = []
    for i_model in range(n_model):
        model = conf.get_model(**task.models[i_model])
        pairs.append((f'model_{i_model}', [model['path'] + '<br/>' + str(model['params'])]))
    rp.append(pairs_to_table(pairs))

    rp.append(Elm('h3', 'Loss per Sample'))
    def get_base_df():
        return pd.DataFrame({'cols_org': info['data']['cols_org']}, index=info['data']['cols'])
    df_ = get_base_df()
    for i_model in range(n_model):
        df_[f'model_{i_model}'] = info['loss_per_sample'][i_model]
    d = {}
    for col in df_.columns:
        d[col] = '' if (col == 'cols_org') else df_[col].mean()
    df_ = pd.concat([df_, pd.DataFrame(d, index=['mean'])])
    for i_model in range(1, n_model):
        df_[f'model_{i_model}-model_0'] = df_[f'model_{i_model}'] - df_[f'model_0']
    rp.append(df_.to_html())

    for i_model in range(n_model):
        rp.append(Elm('h4', f'model_{i_model}'))
        df_ = get_base_df()
        for i_pct, pct in enumerate(task.percentile_points):
            df_[f'{pct:.0%}'] = info['percentiles'][i_model][i_pct]
        for i_pct in range(int(len(task.percentile_points) / 2)):
            pct_0 = task.percentile_points[i_pct]
            pct_1 = task.percentile_points[- (i_pct + 1)]
            df_[f'{pct_1:.0%}-{pct_0:.0%}'] = df_[f'{pct_1:.0%}'] - df_[f'{pct_0:.0%}']
        rp.append(df_.to_html())

    tsta = list(np.load(os.path.join(conf.log_dir, f'sample_0_tsta_task_{i_task}.npy')))
    tsta = [tsta_.replace(':00:00', '') for tsta_ in tsta]
    true = np.load(os.path.join(conf.log_dir, f'sample_0_true_task_{i_task}.npy'))
    preds = [
        np.load(os.path.join(conf.log_dir, f'sample_0_model_{i_model}_task_{i_task}.npy'))
        for i_model in range(n_model)
    ]
    pred_len, n_channel = true.shape
    for i_channel in range(n_channel):
        desc = OrderedDict()
        fig, ax = plt.subplots(nrows=1, figsize=(6, 2))
        x = list(range(pred_len))
        desc['true'] = ax.plot(x, [y_[i_channel] for y_ in true])[0]
        for i_model in range(n_model):
            desc[f'model_{i_model}'] = ax.plot(x, [y_[i_channel] for y_ in preds[i_model]])[0]
        ax.set_xticks(x[::12])
        ax.set_xticklabels(tsta[::12], rotation=90, fontsize=11)
        ax.grid(axis='both', linestyle='dotted', linewidth=1)
        ax.legend(desc.values(), desc.keys(), loc='upper left', bbox_to_anchor=(1.05, 1))
        _add_picture(conf.log_dir, rp, f'task_{i_task}_ts_{i_channel}.png')
        rp.append('<br/>')
    #for i, key_loss in enumerate(li_key_loss):
    #    desc[key_loss] = ax.plot(x, li_loss[i])[0]
    #ax.set_xlabel('i_epoch')
    #ax.set_ylabel(ylabel)


def _report_task(rp, conf, i_task):
    task = conf.tasks[i_task]
    log_path = os.path.join(conf.log_dir, f'info_task_{i_task}.toml')
    info = None if (not os.path.isfile(log_path)) else toml.load(log_path)
    rp.append(Elm('h2', f'task_{i_task} ({task.task_type})'))
    if task.task_type == 'train':
        _report_task_train(rp, conf, i_task, info)
    elif task.task_type == 'eval':
        _report_task_eval(rp, conf, i_task, info)


def report(conf_file):
    conf = Config.from_conf_file(conf_file)

    rp = shirotsubaki.report.Report()
    rp.style.set('body', 'width', '1200px')
    rp.style.set('table', 'margin', '0.5em 0')
    rp.set('title', 'Report')
    rp.append(Elm('h1', conf.out_dir_name))

    for i_task in range(len(conf.tasks)):
        _report_task(rp, conf, i_task)

    out_path = os.path.join(conf.log_dir, 'report.html')
    rp.output(out_path)
    print(out_path)
