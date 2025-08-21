import base64
import os
import re


def copy(target, file_0, file_1):
    with (
        open(file_0, mode='r') as ifile, 
        open(file_1, mode='w', encoding='utf-8', newline='\n') as ofile,
    ):
        for line in ifile:
            match = re.search(r'<img src="((?!data:)[^"]+\.png)"', line)
            if not match:
                ofile.write(line)
                continue
            img_path = os.path.join(target, match.group(1))
            with open(img_path, 'rb') as image_file:
                base64_img = base64.b64encode(image_file.read()).decode('utf-8')
            line_ = line.replace(match.group(1), f'data:image/png;base64, {base64_img}')
            ofile.write(line_)


def main():
    ofile = open('docs/index.md', mode='w', encoding='utf-8', newline='\n')
    targets = [
        'outputs/sample_traffic_mini_0',
    ]
    for target in targets:
        file_0 = os.path.join(target, 'report.html')
        file_1 = os.path.basename(target) + '.html'
        path_1 = os.path.join('docs/', file_1)
        if os.path.isfile(file_0):
            copy(target, file_0, path_1)
        elif not os.path.isfile(path_1):
            assert False, 'No such file: ' + file_0
        ofile.write(f'- [{file_1}]({file_1})\n')
    ofile.close()


if __name__ == '__main__':
    main()
