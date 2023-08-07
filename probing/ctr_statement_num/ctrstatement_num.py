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
    file_cnt = 0
    bar = tqdm.tqdm(file_path_list)
    with open("ctrstatement_num.txt", 'w') as fp:
        for path in bar:
            fname = path.split('/')[-1].split('.')[0]
            dot_path =  '../ast_dot/' + fname + '/1-ast.dot'
            if not os.path.exists(dot_path):
                continue
            fdata = open(dot_path, 'r').read().split('\n')
            # print(fdata[1])
            # print(fdata[3])
            ctr_count = []
            for line in fdata:
                num = line.split('" [')[0].split('"')[-1]
                node_data = re.search(r'<.*</SUB>>', line)
                if node_data != None and num != None:
                    node_data = node_data.group()
                #    print(num)
                #    print(node_data)
                    if_iden = re.search(r'\(CONTROL_STRUCTURE,', node_data)
                    if if_iden != None:
                        ctr_statement = re.search(r'\(.*\)', node_data).group().split(',')[1]
                        # print(ctr_statement)
                        if ctr_statement not in ctr_count:
                            ctr_count.append(ctr_statement)
            # print(len(ctr_count))
            if len(ctr_count) <= 19:
                file_cnt += 1
                fp.write(fname + '.c  ' + str(len(ctr_count)) + '\n')
            bar.update()
    bar.close()
    print(file_cnt)