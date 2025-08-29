from dataclasses import dataclass
from shiroinu.config import Config
import shirotsubaki.report
from shirotsubaki.utils import figure_to_html, style_top_ranks_per_row
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
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Verdana'
plt.rcParams['font.size'] = 11


def module_to_str(module, newline=False):
    if newline and len(module['params']) > 0:
        return f'{module["path"]}<br/>{module["params"]}'
    return f'{module["path"]} {module["params"]}'


class TaskInfoExtractor:
    def model_to_elm(self, model):
        path = model['path'].replace('shiroinu.models.', '')
        elm = Elm('div', path)
        elm.append(Elm('ul'))
        for k, v in model['params'].items():
            v_ = v
            if k == 'state_path':
                v_ = v_.replace(self.conf.log_dir, '')
            elm.inner[-1].append(Elm('li', f'{k}: {v_}'))
        return elm

    def __init__(self, conf, i_task):
        self.id = f'task_{i_task}'
        self.conf = conf
        self.task = conf.tasks[i_task]
        self.name = f'task_{i_task}: {self.task.task_type}'
        log_path = self.conf.log_dir_path(f'info_{self.id}.toml')
        self.info = None if (not os.path.isfile(log_path)) else toml.load(log_path)

        if self.task.task_type == 'train':
            self.model_setting = self.conf.get_model(**self.task.model)
            self.is_target_criteria_0 = (
                (self.task.criterion_target['path'] == self.conf.criteria[0]['path'])
                and (self.task.criterion_target['params'] == self.conf.criteria[0]['params'])
            )
        if self.task.task_type == 'eval':
            self.n_model = len(self.task.models)
            self.model_rows = []
            self.model_names = []
            for i_model in range(self.n_model):
                model = self.conf.get_model(for_report=True, **self.task.models[i_model])
                self.model_rows.append([model['name'], self.model_to_elm(model)])
                self.model_names.append(model['name'])

    def get_windows(self, key):
        n_sample = self.info[key]['n_sample']
        return pd.DataFrame({
            'sample 0': self.info[key]['sample_first'],
            f'sample {n_sample - 1}': self.info[key]['sample_last'],
        }, index=[
            'Start of Input Window', 'End of Input Window',
            'Start of Prediction Window', 'End of Prediction Window']).T.to_html()

    def get_channels(self, key):
        data = {
            'cols_org': self.info[key]['cols_org'],
            'means': self.info[key]['means'],
            'stds': self.info[key]['stds'],
        }
        if 'q1s' in self.info[key]:
            data['q1s'] = self.info[key]['q1s']
            data['q2s'] = self.info[key]['q2s']
            data['q3s'] = self.info[key]['q3s']
        return pd.DataFrame(data, index=self.info[key]['cols']).to_html()

    def get_task_train_conf(self):
        rows = [
            ['criterion_target', module_to_str(self.task.criterion_target)],
            *[[f'criteria {i}', module_to_str(c)] for i, c in enumerate(self.conf.criteria)],
            *[[f'{tp}_range', getattr(self.task, f'{tp}_range')] for tp in ['train', 'valid']],
            ['model_path', self.model_setting['path']],
            ['model_params', self.model_setting['params']],
            *[
                [key, f'{getattr(self.task, key).path} {getattr(self.task, key).params}']
                for key in ['batch_sampler', 'optimizer', 'lr_scheduler']
            ],
        ]
        return Elm.table_from_rows(rows, index=True)

    def plot_loss_graph(self, li_key_loss, ylabel):
        desc = OrderedDict()
        fig, ax = plt.subplots(nrows=1, figsize=(3.5, 2.5))
        li_loss = [[epoch[key_loss] for epoch in self.info['epochs']] for key_loss in li_key_loss]
        x = [i for i in range(len(li_loss[0]))]
        for i, key_loss in enumerate(li_key_loss):
            key_loss_ = 'train' if key_loss.endswith('_train') else 'valid'
            desc[key_loss_] = ax.plot(x, li_loss[i])[0]
        ax.set_xlabel('i_epoch')
        ax.set_ylabel(ylabel)
        ax.grid(axis='both', linestyle='dotted', linewidth=1)
        ax.legend(desc.values(), desc.keys(), loc='upper right')
        return fig

    def get_loss_valid_best(self):
        i_epoch = self.info['epoch_id_best']
        rows = [
            ['epoch_id_best', i_epoch],
            ['loss_valid_best', self.info['epochs'][i_epoch]['loss_0_per_sample_valid']],
        ]
        return Elm.table_from_rows(rows, index=True)

    def get_task_eval_conf(self):
        rows = [
            ['criterion', module_to_str(self.task.criterion_eval)],
            ['valid_range', getattr(self.task, 'valid_range')],
        ]
        return Elm.table_from_rows(rows, index=True)

    def get_models(self):
        return Elm.table_from_rows(self.model_rows, index=True)

    def get_df_loss_per_sample(self):
        df_ = pd.DataFrame(
            {'cols_org': self.info['data']['cols_org']},
            index=self.info['data']['cols'],
        )
        for i_model in range(self.n_model):
            df_[self.model_names[i_model]] = self.info['loss_per_sample'][i_model]
        d = {}
        for col in df_.columns:
            d[col] = '' if (col == 'cols_org') else df_[col].mean()
        df_ = pd.concat([pd.DataFrame(d, index=['mean']), df_])
        for i_model in range(1, self.n_model):
            df_[f'{self.model_names[0]}-{self.model_names[i_model]}'] \
                = df_[self.model_names[0]] - df_[self.model_names[i_model]]
        return df_

    def plot_prediction(
        self, ax, x, true, preds, i_channel, tsta,
        xtick_step=12, show_xticklabels=True, diff=False,
    ):
        colname = self.info['data']['cols'][i_channel]
        colname_org = self.info['data']['cols_org'][i_channel]

        desc = OrderedDict()
        li_true = [y_[i_channel] for y_ in true]
        if diff:
            desc['true'] = ax.plot(x, [0.0 for _ in true])[0]        
        else:
            desc['true'] = ax.plot(x, li_true)[0]
        for i_model in range(self.n_model):
            li_pred = [y_[i_channel] for y_ in preds[i_model]]
            if diff:
                li_pred = [y_1 - y_0 for y_0, y_1 in zip(li_true, li_pred)]
            desc[self.model_names[i_model]] = ax.plot(x, li_pred)[0]
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
            ncol=math.ceil((self.n_model + 1) / 4.0),
            labelspacing=0.2,
        )

    def plot_predictions(
        self, tsta, true, preds, prefix,
        _figure_to_html, diff=False, max_n_graph=200,
    ):
        pred_len, n_channel = true.shape
        x = list(range(pred_len))
        contents = {}
        for i_graph, i_channel_0 in enumerate(range(0, n_channel, 5)):
            li_i_channel = list(range(i_channel_0, n_channel))[:5]
            n_channel_ = len(li_i_channel)
            if n_channel_ == 1:
                fig, ax = plt.subplots(nrows=1, figsize=(6.5, 1.3))
                self.plot_prediction(ax, x, true, preds, i_channel_0, tsta, diff=diff)
            else:
                fig, ax = plt.subplots(nrows=n_channel_, figsize=(6.5, 1.3 * n_channel_))
                for i_ax, i_channel in enumerate(li_i_channel):
                    self.plot_prediction(
                        ax[i_ax], x, true, preds, i_channel, tsta,
                        diff=diff, show_xticklabels=(i_ax == n_channel_ - 1),
                    )
                plt.subplots_adjust(hspace=0.1)
            contents[f'y{i_channel_0}-'] = _figure_to_html(fig, f'img/{prefix}_{i_channel_0}.svg')
            if (i_graph + 1) >= max_n_graph:
                break
        return contents


@dataclass(frozen=False)
class ReportWriter:
    rp: shirotsubaki.report.Report
    conf: Config
    image_format: str = 'svg'
    embed_image: bool = True
    dpi: int = 100
    max_n_graph: int = 200

    def append(self, content):
        self.rp.append(content)

    def append_as_minitabs(self, tabs_id, contents):
        self.rp.append_as_minitabs(tabs_id, contents, tabs_per_line=20)

    def _figure_to_html(self, fig, img_rel_path):
        return figure_to_html(
            fig, fmt=self.image_format, embed=self.embed_image,
            html_dir=self.conf.log_dir, img_rel_path=img_rel_path,
            dpi=self.dpi, callback=plt.close,
        )

    def append_figure(self, fig, img_rel_path):
        self.append(self._figure_to_html(fig, img_rel_path))

    def _report_task_train(self, tie):
        self.append(tie.get_task_train_conf())
        for type_ in ['train', 'valid']:
            key = f'data_{type_}'
            self.append(Elm('h3', key))
            self.append(tie.get_windows(key))
            if type_ == 'train':
                self.rp.append_as_toggle(f'{tie.id}_{key}_channels', tie.get_channels(key))
        self.append(tie.get_loss_valid_best())

        for i_crit, crit in enumerate(self.conf.criteria):
            if i_crit == 0:
                if tie.is_target_criteria_0:
                    fig = tie.plot_loss_graph(
                        ['loss_0_per_sample_train', 'loss_0_per_sample_valid'],
                        tie.task.criterion_target['path'])
                    self.append_figure(fig, f'img/{tie.id}_loss_train_valid.svg')
                    continue
                fig = tie.plot_loss_graph(['loss_0_per_sample_train'], tie.task.criterion_target['path'])
                self.append_figure(fig, f'img/{tie.id}_loss_train.svg')
            fig = tie.plot_loss_graph([f'loss_{i_crit}_per_sample_valid'], crit['path'])
            self.append_figure(fig, f'img/{tie.id}_loss_{i_crit}_valid.svg')

    def _report_task_eval(self, tie):
        self.append(tie.get_task_eval_conf())
        self.append(Elm('h3', 'Models'))
        self.append(tie.get_models())
        self.append(Elm('h3', 'data'))
        self.append(tie.get_windows(f'data'))
        self.rp.append_as_toggle(f'{tie.id}_data_channels', tie.get_channels(f'data'))

        self.append(Elm('h3', 'Loss per Sample'))
        df_ = tie.get_df_loss_per_sample()
        self.append(style_top_ranks_per_row(df_.head(1).style, tie.model_names).to_html())
        content = style_top_ranks_per_row(df_.style, tie.model_names).to_html()
        self.rp.append_as_toggle(f'{tie.id}_loss_per_sample', content)

        tsta = list(np.load(self.conf.log_dir_path(f'sample_0_tsta_{tie.id}.npy')))
        tsta = [tsta_.replace(':00', '') for tsta_ in tsta]
        true = np.load(self.conf.log_dir_path(f'sample_0_true_{tie.id}.npy'))
        preds = [
            np.load(self.conf.log_dir_path(f'sample_0_model_{i_model}_{tie.id}.npy'))
            for i_model in range(tie.n_model)
        ]

        self.append(Elm('h3', 'Prediction Plot'))
        prefix = f'{tie.id}_pred'
        contents = tie.plot_predictions(
            tsta, true, preds,
            prefix, self._figure_to_html, max_n_graph=self.max_n_graph,
        )
        self.append_as_minitabs(prefix, contents)

        self.append(Elm('h3', 'Prediction Plot (Diff)'))
        prefix = f'{tie.id}_pred_diff'
        contents = tie.plot_predictions(
            tsta, true, preds,
            prefix, self._figure_to_html, diff=True, max_n_graph=self.max_n_graph,
        )
        self.append_as_minitabs(prefix, contents)

    def _report_task(self, tie):
        self.append(Elm('h2', tie.name))
        if tie.task.task_type == 'train':
            self._report_task_train(tie)
        elif tie.task.task_type == 'eval':
            self._report_task_eval(tie)

    def summary(self):
        elm = Elm('div').set_attr('id', 'summary')
        wl = self.conf.data.get('white_list', '')
        if wl != '':
            if isinstance(wl, list):
                count = len(wl)
            else:
                count = len(wl.split(','))
            wl = f'count: {count}<br/>{wl}'
        tbl = Elm.table_from_rows([
            ['data path', self.conf.data['path'].replace(os.path.expanduser('~/'), '~/')],
            ['white_list', wl],
            ['seq_len', self.conf.data['seq_len']],
            ['pred_len', self.conf.data['pred_len']],
        ], index=True).set_attr('style', 'max-width: 720px;')
        elm.append(tbl)
        rows = [['', 'train_range', 'valid_range', 'criterion', 'model', 'n_epoch', 'loss_valid']]
        for i_task in range(len(self.conf.tasks)):
            tie = TaskInfoExtractor(self.conf, i_task)
            crit_key = 'criterion_' + ('target' if (tie.task.task_type) == 'train' else 'eval')
            model = ''
            if tie.task.task_type == 'train':
                model = tie.model_setting['path'].replace('shiroinu.models.', '')
            else:
                model = '<br/>'.join(tie.model_names)
            loss = ''
            if tie.task.task_type == 'train':
                loss_ = tie.info['epochs'][tie.info['epoch_id_best']]['loss_0_per_sample_valid']
                loss = f'{loss_:.5f}'
            else:
                losses = tie.get_df_loss_per_sample().iloc[0]
                loss = '<br/>'.join([f'{losses[model_name]:.5f}' for model_name in tie.model_names])
            rows.append([
                tie.name,
                getattr(tie.task, 'train_range', ''),
                getattr(tie.task, 'valid_range'),
                module_to_str(getattr(tie.task, crit_key), newline=True),
                model,
                '' if tie.task.task_type == 'eval' else str(tie.info['epoch_id_best']),
                loss,
            ])
        elm.append(Elm.table_from_rows(rows, index=True, header=True))
        self.append(elm)

    def run(self):
        self.rp.style.set('body', 'min-width', '1350px')
        self.rp.style.set('ul', 'margin', '0')
        self.rp.style.set('ul', 'padding-left', '2em')
        self.rp.style.set('th, td', 'vertical-align', 'top')

        self.rp.set('title', self.conf.out_dir_name)
        self.append(Elm('h1', self.conf.out_dir_name))

        self.append(Elm('h2', 'summary'))
        self.summary()

        for i_task in range(len(self.conf.tasks)):
            tie = TaskInfoExtractor(self.conf, i_task)
            self._report_task(tie)
        out_path = self.conf.log_dir_path('report.html')

        self.rp.output(out_path)
        self.rp = None

    @classmethod
    def create(cls, conf_file, image_format, embed_image, dpi, max_n_graph):
        return cls(
            rp=shirotsubaki.report.Report(),
            conf=Config.from_conf_file(conf_file),
            image_format=image_format,
            embed_image=embed_image,
            dpi=dpi,
            max_n_graph=max_n_graph,
        )

    def __call__(self) -> None:
        return self.run()
