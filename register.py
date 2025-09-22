import base64
import os
import re
import shirotsubaki.report
from shirotsubaki.element import Element as Elm
from bs4 import BeautifulSoup
from pathlib import Path


def _copy(target, file_0, file_1):
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


def _get_path(target):
    return (
        os.path.basename(target),
        os.path.join(target, 'report.html'),
        os.path.basename(target) + '.html',
        os.path.join('docs/', os.path.basename(target) + '.html')
    )


def _to_a(base, file_1):
    a = Elm('a', base).set_attr('href', file_1)
    return a.set_attr('target', '_blank').set_attr('rel', 'noopener noreferrer')


def main():
    targets = {
        'Traffic': [
            'outputs/sample_traffic_sa',
            'outputs/sample_traffic_dlinear_0',
        ],
        'Weather': [
            'outputs/sample_weather_sa',
        ],
        'For Debugging': [
            'outputs/sample_traffic_mini_0',
            'outputs/sample_traffic_mini_1',
            'outputs/sample_traffic_mini_2',
        ],
    }

    for k, v in targets.items():
        for target in v:
            _, file_0, file_1, path_1 = _get_path(target)
            if os.path.isfile(file_0):
                _copy(target, file_0, path_1)
            elif not os.path.isfile(path_1):
                assert False, 'No such file: ' + file_0

    rp = shirotsubaki.report.Report()
    rp.style.set('ul', 'margin', '0')
    rp.style.set('ul', 'padding-left', '2em')
    rp.style.set('th, td', 'vertical-align', 'top')

    for k, v in targets.items():
        rp.append(Elm('h2', k))
        ul = Elm('ul')
        for target in v:
            base, _, file_1, _ = _get_path(target)
            ul.append(Elm('li', Elm('a', base).set_attr('href', f'#{base}')))
        rp.append(ul)
    rp.append(Elm('br'))
    rp.append(Elm('hr'))

    for k, v in targets.items():
        rp.append(Elm('h2', k))
        for target in v:
            base, _, file_1, path_1 = _get_path(target)
            text = Path(path_1).read_text(encoding='utf-8')
            soup = BeautifulSoup(text, 'html.parser')
            node = soup.select_one('#summary')
            if node:
                rp.append(Elm('h3', _to_a(base, file_1)).set_attr('id', base))
                rp.append(''.join(str(c) for c in node.contents))

    rp.output('docs/index.html')


if __name__ == '__main__':
    main()
