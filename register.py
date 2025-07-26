import shutil


def main():
    ofile = open('index.md', mode='w', encoding='utf-8', newline='\n')
    targets = [
        'outputs/sample_mini',
    ]
    results = []
    for target in targets:
        file_0 = target + '/report.html'
        file_1 = target.split('/')[-1] + '.html'
        shutil.copy(file_0, file_1)
        ofile.write(f'- [{file_1}]({file_1})\n')
        results.append(file_1)
    ofile.close()


if __name__ == '__main__':
    main()
