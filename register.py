import base64
import os
import re
import shirotsubaki.report
from shirotsubaki.element import Element as Elm
from bs4 import BeautifulSoup
from pathlib import Path
import argparse


def _copy(target, file_0, file_1):
    # TODO: SVG ファイルがセパレートされていたときの埋め込み化に対応していません
    with (
        open(file_0, mode='r', encoding='utf-8') as ifile, 
        open(file_1, mode='w', encoding='utf-8', newline='\n') as ofile,
    ):
        for i, line in enumerate(ifile):
            match = re.search(r'<img src="((?!data:)[^"]+\.png)"', line)
            if not match:
                ofile.write(line)
                continue
            img_path = os.path.join(target, match.group(1))
            with open(img_path, 'rb') as image_file:
                base64_img = base64.b64encode(image_file.read()).decode('utf-8')
            line_ = line.replace(match.group(1), f'data:image/png;base64, {base64_img}')
            ofile.write(line_)


def _get_paths(dir):
    return {
        'dirname': os.path.basename(dir),
        'report_path_org': os.path.join(dir, 'report.html'),
        'report_path_target': os.path.join('docs/', os.path.basename(dir) + '.html'),
        'report_rel_path_target': os.path.basename(dir) + '.html',
    }


def _to_a(s, rel_path):
    a = Elm('a', s).set_attr('href', rel_path)
    return a.set_attr('target', '_blank').set_attr('rel', 'noopener noreferrer')


def main():
    targets = {
        'Traffic': [
            {'dir': 'outputs/sample_traffic_sa'},
            {'dir': 'outputs/sample_traffic_dlinear_0'},
        ],
        'Weather': [
            {'dir': 'outputs/sample_weather_sa'},
        ],
        'For Debugging': [
            {'dir': 'outputs/sample_traffic_tiny_0'},
            {'dir': 'outputs/sample_traffic_mini_0'},
            {'dir': 'outputs/sample_traffic_mini_1'},
            {'dir': 'outputs/sample_traffic_mini_2'},
        ],
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('target_path', type=str, nargs='?', default='sample_traffic_mini_0')
    args = parser.parse_args()

    flag = False
    for category, reports in targets.items():
        for report in reports:
            if report['dir'] == f'outputs/{args.target_path}':
                print('Target: ' + report['dir'])
                report['collect'] = True
                flag = True
                break
    if not flag:
        assert False, 'Invalid target: ' + args.target_path

    for category, reports in targets.items():
        for report in reports:
            report.update(_get_paths(report['dir']))
            if ('collect' not in report) or (not report['collect']):
                continue
            if os.path.isfile(report['report_path_org']):
                _copy(report['dir'], report['report_path_org'], report['report_path_target'])
            else:
                assert False, 'No such file: ' + report['report_path_org']

    rp = shirotsubaki.report.Report()
    rp.style.set('ul', 'margin', '0')
    rp.style.set('ul', 'padding-left', '2em')
    rp.style.set('th, td', 'vertical-align', 'top')

    for category, reports in targets.items():
        rp.append(Elm('h2', category))
        ul = Elm('ul')
        for report in reports:
            dirname = report['dirname']
            ul.append(Elm('li', Elm('a', dirname).set_attr('href', f'#{dirname}')))
        rp.append(ul)
    rp.append(Elm('br'))
    rp.append(Elm('hr'))

    for category, reports in targets.items():
        rp.append(Elm('h2', category))
        for report in reports:
            text = Path(report['report_path_target']).read_text(encoding='utf-8')
            soup = BeautifulSoup(text, 'html.parser')
            node = soup.select_one('#summary')
            if node:
                elm = Elm('h3', _to_a(report['dirname'], report['report_rel_path_target']))
                elm.set_attr('id', report['dirname'])
                rp.append(elm)
                rp.append(''.join(str(c) for c in node.contents))

    rp.output('docs/index.html')


if __name__ == '__main__':
    main()
