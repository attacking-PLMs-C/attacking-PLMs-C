import os
import tqdm
import re
import operator
from collections import Counter

def get_file_path(root_path, file_list):
    PATH = os.listdir(root_path)
    for path in PATH:
        # print(path)
        co_path = os.path.join(root_path, path)
        if os.path.isfile(co_path):
            file_list.append(co_path)
        elif os.path.isdir(co_path):
            get_file_path(co_path, file_list)
    return file_list

if __name__ == "__main__":
    file_path_list = []
    get_file_path('../valid_code', file_path_list)
    count = 0
    file_cnt = 0
    bar = tqdm.tqdm(file_path_list)
    with open("identifier_num.txt", 'w') as fp:
        for path in bar:
            fname = path.split('/')[-1].split('.')[0]
            dot_path =  '../ast_dot/' + fname + '/1-ast.dot'
            if not os.path.exists(dot_path):
                continue
            # file_cnt += 1
            fdata = open(dot_path, 'r').read().split('\n')
            # print(fdata[1])
            # print(fdata[3])
            iden_count = []
            for line in fdata:
                # print(line)
                num = line.split('" [')[0].split('"')[-1]
                node_data = re.search(r'<.*</SUB>>', line)
                if node_data != None and num != None:
                    node_data = node_data.group()
                #    print(num)
                    # print(node_data)
                    if_iden = re.search(r'\(IDENTIFIER,', node_data)
                    if if_iden != None:
                        iden_content = re.search(r'\(.*\)', node_data).group().split(',')[1]
                        # print(iden_content)
                        if iden_content not in iden_count:
                            iden_count.append(iden_content)
            if len(iden_count) <= 39:
                count += 1
                fp.write(fname + '.c  ' + str(len(iden_count)) + '\n')
            # print(iden_count)
            # print(len(iden_count))
            # break
            # print(ide_count)
            bar.update()
    bar.close()
    print(count)
    # print(file_cnt)
    # cnt = 0
    # with open('identifier_num_labels.txt', 'w') as fp:
    #     with  open('identifier_num.txt', 'r') as fp1:
    #         fdata = fp1.read().split('\n')
    #         for line in fdata:
    #             fname = line.split('  ')[0]
    #             fiden_num = int(line.split('  ')[-1])
    #             if fiden_num <= 39:
    #                 cnt += 1
    #                 fp.write(fname + '  ' + str(fiden_num) + '\n')
    #             # if fiden_num >= 0 and fiden_num < 4:
    #             #     fp.write(fname + '  0\n')
    #             # elif fiden_num >= 4 and fiden_num < 8:
    #             #     fp.write(fname + '  1\n')
    #             # elif fiden_num >= 8 and fiden_num < 12:
    #             #     fp.write(fname + '  3\n')
    #             # elif fiden_num >= 12 and fiden_num < 16:
    #             #     fp.write(fname + '  4\n')
    #             # elif fiden_num >= 16 and fiden_num < 20:
    #             #     fp.write(fname + '  5\n')
    #             # elif fiden_num >= 20 and fiden_num < 24:
    #             #     fp.write(fname + '  6\n')
    #             # elif fiden_num >= 24 and fiden_num < 28:
    #             #     fp.write(fname + '  7\n')
    #             # elif fiden_num >= 28 and fiden_num < 32:
    #             #     fp.write(fname + '  8\n')
    #             # elif fiden_num >= 32 and fiden_num <=36:
    #             #     fp.write(fname + '  9\n')
    # print(cnt)