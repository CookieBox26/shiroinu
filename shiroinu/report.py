from shiroinu.config import Config
import shirotsubaki.report
from shirotsubaki.element import Element as Elm
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import base64
import io
import types
import toml
import os
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11


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
    fig, ax = plt.subplots(nrows=1, figsize=(4, 3))
    desc = OrderedDict()
    li_loss = []
    for key_loss in li_key_loss:
        li_loss.append([epoch[key_loss] for epoch in info['epochs']])
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
        rp.append(_get_channels_df(info, f'data_{type_}').to_html())
        rp.append(_get_ranges_df(conf, info, f'data_{type_}').to_html())

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
    for i_, criteriton in enumerate(conf.criteria[1:]):
        i = i_ + 1
        _plot_loss_graph(info, [f'loss_{i}_per_sample_valid'], conf.criteria[1]['path'])
        _add_picture(conf.log_dir, rp, f'task_{i_task}_loss_{i}_valid.png')


def _report_task(rp, conf, i_task):
    task = conf.tasks[i_task]
    log_path = os.path.join(conf.log_dir, f'info_task_{i_task}.toml')
    info = None if (not os.path.isfile(log_path)) else toml.load(log_path)
    rp.append(Elm('h2', f'task_{i_task} ({task.task_type})'))
    if task.task_type == 'train':
        _report_task_train(rp, conf, i_task, info)


def report(conf_file):
    conf = Config.from_conf_file(conf_file)

    rp = shirotsubaki.report.Report()
    rp.style.set('body', 'width', '1200px')
    rp.style.set('table', 'margin-bottom', '0.5em')
    def append(self, v):
        self.append_to('content', v)
    rp.append = types.MethodType(append, rp)
    rp.set('title', 'Report')
    rp.append(Elm('h1', conf.out_dir_name))

    for i_task in range(len(conf.tasks)):
        _report_task(rp, conf, i_task)

    out_path = os.path.join(conf.log_dir, 'report.html')
    rp.output(out_path)
