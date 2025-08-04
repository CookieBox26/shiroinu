from shiroinu.config import Config
import shirotsubaki.report
from shirotsubaki.element import Element as Elm
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from collections import OrderedDict
import base64
import io
import toml
import os
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11


def _append_as_toggle(rp, toggle_id, content):
    rp.style.set(f'#toggle-{toggle_id}', 'display', 'none')
    rp.style.set(f'.content-{toggle_id}', 'display', 'none')
    rp.style.set(f'#toggle-{toggle_id}:checked ~ .content-{toggle_id}', 'display', 'block')
    rp.append(Elm('label', 'Show details').set_attr('for', f'toggle-{toggle_id}'))
    rp.append(Elm('input').set_attr('type', 'checkbox').set_attr('id', f'toggle-{toggle_id}'))
    rp.append(Elm('div', content).set_attr('class', f'toggle-area content-{toggle_id}'))


def _append_as_tabs(rp, tabs_id, contents):
    rp.style.set(', '.join([
        f'#{tabs_id}-btn{i_tab:04d}:checked ~ #{tabs_id}-content{i_tab:04d}'
        for i_tab in range(len(contents))
    ]), 'display', 'block')
    rp.style.set(', '.join([
        f':has(#{tabs_id}-btn{i_tab:04d}:checked) label[for="{tabs_id}-btn{i_tab:04d}"]'
        for i_tab in range(len(contents))
    ]), 'background', '#c9c9c9')
    for i_tab, tab_name in enumerate(contents.keys()):
        id_ = f'{tabs_id}-btn{i_tab:04d}'
        checked = 'checked' if (i_tab == 0) else ''
        label = Elm('label', tab_name).set_attr('class', 'tabbtn')
        label.set_attr('for', id_)
        rp.append(label)
        rp.append(f'<input type="radio" name="{tabs_id}" id="{id_}" hidden {checked}/>')
        if (i_tab + 1) % 20 == 0:
            rp.append('<br/>')
    for i_tab, content in enumerate(contents.values()):
        id_ = f'{tabs_id}-content{i_tab:04d}'
        rp.append(Elm('div', content).set_attr('id', id_).set_attr('class', 'tab-content'))


def _pairs_to_table(pairs):
    tbody = Elm('tbody')
    for pair in pairs:
        tbody.append(Elm('tr'))
        tbody.inner[-1].append(Elm('td', f'<b>{pair[0]}</b>'))
        for v in pair[1]:
            tbody.inner[-1].append(Elm('td', f'{v}'))
    return Elm('table', tbody).set_attr('border', '1')


def _to_picture(output_path, filename, emb=False):
    kwargs = {'format': 'png', 'bbox_inches': 'tight'}  # , 'dpi': 72
    if emb:
        pic_io_bytes = io.BytesIO()
        plt.savefig(pic_io_bytes, **kwargs)
        pic_io_bytes.seek(0)
        base64_img = base64.b64encode(pic_io_bytes.read()).decode('utf8')
        src = f'data:image/png;base64, {base64_img}'
    else:
        img_dir = os.path.join(output_path, 'img/')
        os.makedirs(img_dir, exist_ok=True)
        plt.savefig(f'{img_dir}{filename}', **kwargs)
        src = f'img/{filename}'
    plt.close()
    return f'<img src="{src}"/>'


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
    data = {
        'cols_org': info[key]['cols_org'],
        'means': info[key]['means'],
        'stds': info[key]['stds'],
    }
    if 'q1s' in info[key]:
        data['q1s'] = info[key]['q1s']
        data['q2s'] = info[key]['q2s']
        data['q3s'] = info[key]['q3s']
    return pd.DataFrame(data, index=info[key]['cols'])


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


def _report_task_train(rp, conf, i_task, info, embed_image):
    task = conf.tasks[i_task]

    for type_ in ['train', 'valid']:
        rp.append(Elm('h3', f'data_{type_}'))
        path = conf.data['path'].replace(os.path.expanduser('~/'), '~/')
        rp.append(_pairs_to_table([('path', [path])]))
        rp.append(_get_ranges_df(conf, info, f'data_{type_}').to_html())
        content = _get_channels_df(info, f'data_{type_}').to_html()
        if type_ == 'train':
            _append_as_toggle(rp, f'toggle_task_{i_task}_data_{type_}_channels', content)

    rp.append(Elm('h3', 'Model and Optimizer'))
    model_settings = conf.get_model(**task.model)
    pairs = [
        ('model_path', [model_settings['path']]),
        ('model_params', [model_settings['params']])]
    rp.append(_pairs_to_table(pairs))
    pairs = [
        (key, [getattr(task, key).path, getattr(task, key).params])
        for key in ['batch_sampler', 'optimizer', 'lr_scheduler']]
    rp.append(_pairs_to_table(pairs))

    if (
        (task.criterion_target['path'] == conf.criteria[0]['path'])
        and (task.criterion_target['params'] == conf.criteria[0]['params'])
    ):
        _plot_loss_graph(
            info, ['loss_0_per_sample_train', 'loss_0_per_sample_valid'],
            task.criterion_target['path'])
        rp.append(_to_picture(conf.log_dir, f'task_{i_task}_loss_train_valid.png', embed_image))
    else:
        _plot_loss_graph(info, ['loss_0_per_sample_train'], task.criterion_target['path'])
        rp.append(_to_picture(conf.log_dir, f'task_{i_task}_loss_train.png', embed_image))
        _plot_loss_graph(info, ['loss_0_per_sample_valid'], conf.criteria[0]['path'])
        rp.append(_to_picture(conf.log_dir, f'task_{i_task}_loss_0_valid.png', embed_image))
    for i_crit in range(1, len(conf.criteria)):
        _plot_loss_graph(info, [f'loss_{i_crit}_per_sample_valid'], conf.criteria[i_crit]['path'])
        rp.append(_to_picture(conf.log_dir, f'task_{i_task}_loss_{i_crit}_valid.png', embed_image))

    rp.append(pd.DataFrame({
        'epoch_id_best': [info['epoch_id_best']],
        'loss_valid_best': [info['epochs'][info['epoch_id_best']]['loss_0_per_sample_valid']],
    }).to_html(index=False))


def _plot_prediction(
    ax, model_names, x, true, preds, i_channel, tsta, info,
    xtick_step=12, show_xticklabels=True, diff=False,
):
    n_model = len(model_names)
    colname = info['data']['cols'][i_channel]
    colname_org = info['data']['cols_org'][i_channel]

    desc = OrderedDict()
    li_true = [y_[i_channel] for y_ in true]
    if diff:
        desc['true'] = ax.plot(x, [0.0 for _ in true])[0]        
    else:
        desc['true'] = ax.plot(x, li_true)[0]
    for i_model in range(n_model):
        li_pred = [y_[i_channel] for y_ in preds[i_model]]
        if diff:
            li_pred = [y_1 - y_0 for y_0, y_1 in zip(li_true, li_pred)]
        desc[model_names[i_model]] = ax.plot(x, li_pred)[0]
    ax.set_xticks(x[::xtick_step])
    if show_xticklabels:
        ax.set_xticklabels(tsta[::xtick_step], rotation=90, fontsize=11)
    else:
        ax.tick_params(labelbottom=False)
    ax.set_ylabel(f'{colname} ({colname_org})')
    ax.grid(axis='both', linestyle='dotted', linewidth=1)
    ax.legend(
        desc.values(),
        desc.keys(),
        loc='upper left',
        bbox_to_anchor=(1.01, 1),
        ncol=math.ceil((n_model + 1) / 4.0),
    )


def _plot_predictions(
    rp, model_names, true, preds, tsta, info, output_path, prefix,
    diff=False, embed_image=False, max_n_graph=200,
):
    pred_len, n_channel = true.shape
    x = list(range(pred_len))
    contents = {}
    for i_graph, i_channel_0 in enumerate(range(0, n_channel, 5)):
        li_i_channel = list(range(i_channel_0, n_channel))[:5]
        n_channel_ = len(li_i_channel)
        if n_channel_ == 1:
            fig, ax = plt.subplots(nrows=1, figsize=(7.5, 1.5))
            _plot_prediction(ax, model_names, x, true, preds, i_channel_0, tsta, info, diff=diff)
        else:
            fig, ax = plt.subplots(nrows=n_channel_, figsize=(7.5, 1.5 * n_channel_))
            for i_ax, i_channel in enumerate(li_i_channel):
                _plot_prediction(
                    ax[i_ax], model_names, x, true, preds, i_channel, tsta, info,
                    diff=diff, show_xticklabels=(i_ax == n_channel_ - 1),
                )
            plt.subplots_adjust(hspace=0.1)
        contents[f'y{i_channel_0}-'] = \
            _to_picture(output_path, f'{prefix}_{i_channel_0}.png', embed_image)
        if (i_graph + 1) >= max_n_graph:
            break
    _append_as_tabs(rp, prefix, contents)


def _highlight_min_and_second_min(df, model_names):
    cols_0 = model_names
    cols_1 = [col for col in df.columns if col.startswith(f'{model_names[0]}-')]
    def _highlight_0(row):
        styles = pd.Series('', index=row.index)
        indices = row.sort_values().index
        styles[indices[0]] = 'background: #8dce6f;'
        styles[indices[1]] = 'background: #ccf188;'
        return styles
    def _highlight_1(v):
        return 'color: #d62728;' if v < 0 else 'color: #048243;'
    return df.style.apply(
        _highlight_0, axis=1, subset=cols_0,
    ).map(
        _highlight_1, subset=cols_1,
    )


def _report_task_eval(rp, conf, i_task, info, embed_image, max_n_graph):
    task = conf.tasks[i_task]
    n_model = len(task.models)

    rp.append(Elm('h3', 'data'))
    path = conf.data['path'].replace(os.path.expanduser('~/'), '~/')
    rp.append(_pairs_to_table([('path', [path])]))
    rp.append(_get_ranges_df(conf, info, 'data').to_html())
    content = _get_channels_df(info, 'data').to_html()
    _append_as_toggle(rp, f'toggle_task_{i_task}_data_channels', content)

    rp.append(Elm('h3', 'Criterion'))
    pairs = [('criterion_eval', [task.criterion_eval['path'], task.criterion_eval['params']])]
    rp.append(_pairs_to_table(pairs))

    rp.append(Elm('h3', 'Models'))
    pairs = []
    for i_model in range(n_model):
        model = conf.get_model(for_report=True, **task.models[i_model])
        model_name = f'model_{i_model}' if ('name' not in model) else model['name']
        pairs.append((model_name, [model['path'] + '<br/>' + str(model['params'])]))
    rp.append(_pairs_to_table(pairs))
    model_names = [v[0] for v in pairs]

    rp.append(Elm('h3', 'Loss per Sample'))
    def get_base_df():
        return pd.DataFrame({'cols_org': info['data']['cols_org']}, index=info['data']['cols'])
    df_ = get_base_df()
    for i_model in range(n_model):
        df_[model_names[i_model]] = info['loss_per_sample'][i_model]
    d = {}
    for col in df_.columns:
        d[col] = '' if (col == 'cols_org') else df_[col].mean()
    df_ = pd.concat([pd.DataFrame(d, index=['mean']), df_])
    for i_model in range(1, n_model):
        df_[f'{model_names[0]}-{model_names[i_model]}'] \
            = df_[model_names[0]] - df_[model_names[i_model]]
    rp.append(_highlight_min_and_second_min(df_.head(1), model_names).to_html())
    content = _highlight_min_and_second_min(df_, model_names).to_html()
    _append_as_toggle(rp, f'toggle_task_{i_task}_model_{i_model}_loss', content)

    if False:
        for i_model in range(n_model):
            rp.append(Elm('h4', f'model_{i_model}'))
            df_ = get_base_df()
            for i_pct, pct in enumerate(task.percentile_points):
                df_[f'{pct:.0%}'] = info['percentiles'][i_model][i_pct]
            for i_pct in range(int(len(task.percentile_points) / 2)):
                pct_0 = task.percentile_points[i_pct]
                pct_1 = task.percentile_points[- (i_pct + 1)]
                df_[f'{pct_1:.0%}-{pct_0:.0%}'] = df_[f'{pct_1:.0%}'] - df_[f'{pct_0:.0%}']
            content = df_.to_html()
            _append_as_toggle(rp, f'toggle_task_{i_task}_model_{i_model}_loss_percentiles', content)

    tsta = list(np.load(os.path.join(conf.log_dir, f'sample_0_tsta_task_{i_task}.npy')))
    tsta = [tsta_.replace(':00', '') for tsta_ in tsta]
    true = np.load(os.path.join(conf.log_dir, f'sample_0_true_task_{i_task}.npy'))
    preds = [
        np.load(os.path.join(conf.log_dir, f'sample_0_model_{i_model}_task_{i_task}.npy'))
        for i_model in range(n_model)
    ]

    rp.append(Elm('h3', 'Prediction Plot'))
    _plot_predictions(
        rp, model_names, true, preds, tsta, info, conf.log_dir,
        f'task_{i_task}_pred', embed_image=embed_image, max_n_graph=max_n_graph,
    )

    rp.append(Elm('h3', 'Prediction Plot (Diff)'))
    _plot_predictions(
        rp, model_names, true, preds, tsta, info, conf.log_dir,
        f'task_{i_task}_pred_diff', diff=True, embed_image=embed_image, max_n_graph=max_n_graph,
    )

    #for i, key_loss in enumerate(li_key_loss):
    #    desc[key_loss] = ax.plot(x, li_loss[i])[0]
    #ax.set_xlabel('i_epoch')
    #ax.set_ylabel(ylabel)


def _report_task(rp, conf, i_task, embed_image, max_n_graph):
    task = conf.tasks[i_task]
    log_path = os.path.join(conf.log_dir, f'info_task_{i_task}.toml')
    info = None if (not os.path.isfile(log_path)) else toml.load(log_path)
    rp.append(Elm('h2', f'task_{i_task} ({task.task_type})'))
    if task.task_type == 'train':
        _report_task_train(rp, conf, i_task, info, embed_image)
    elif task.task_type == 'eval':
        _report_task_eval(rp, conf, i_task, info, embed_image, max_n_graph)


def report(conf_file, embed_image, max_n_graph=200):
    conf = Config.from_conf_file(conf_file)
    print('Embed image in report' if embed_image else 'Output image file separately')

    rp = shirotsubaki.report.Report()
    rp.style.set('body', 'min-width', '1350px')
    rp.style.set('table', 'margin-bottom', '0.5em')

    rp.style.set('label', 'cursor', 'pointer')
    rp.style.set('label', 'color', '#016795')
    rp.style.set('.toggle-area', 'background', '#f0f0f0')
    rp.style.set('.toggle-area', 'padding', '1em')
    rp.style.set('.tab-content', 'display', 'none')
    rp.style.set('label.tabbtn', 'color', '#303030')
    rp.style.set('label.tabbtn', 'border', '1px solid #303030')
    rp.style.set('label.tabbtn', 'padding', '0.2em 0.4em')
    rp.style.set('label.tabbtn', 'margin-bottom', '0.4em')
    rp.style.set('label.tabbtn', 'background', '#f0f0f0')
    rp.style.set('label.tabbtn', 'display', 'inline-block')
    rp.style.set('label.tabbtn', 'user-select', 'none')

    rp.set('title', conf.out_dir_name)
    rp.append(Elm('h1', conf.out_dir_name))
    for i_task in range(len(conf.tasks)):
        _report_task(rp, conf, i_task, embed_image, max_n_graph)

    out_path = os.path.join(conf.log_dir, 'report.html')
    rp.output(out_path)
    print(out_path)
