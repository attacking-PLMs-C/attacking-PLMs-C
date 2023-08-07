import re
import os
from collections import Counter
import tqdm
import operator
import random
import json
import pandas as pd
import subprocess
import itertools
from multiprocessing import Pool


def get_file_path(root_path, file_list):
    PATH = os.listdir(root_path)
    for path in PATH:
        co_path = os.path.join(root_path, path)
        if os.path.isfile(co_path):
            file_list.append(co_path)
        elif os.path.isdir(co_path):
            get_file_path(co_path, file_list)
    return file_list

def get_data(code_data):
    code_data = re.sub(r"(\s)+", '', code_data).strip()  # [\n ]+
    # with open("example.txt", 'w') as fp:
    #     fp.write(code_data)
    # code_data = code_data.split(' ')
    return code_data

if __name__ == "__main__":
    total_path_list = []
    mutated_list = []
    noun_mutated_list = []
    get_file_path('../valid_code', total_path_list)
    get_file_path('./mutated_code', mutated_list)
    get_file_path('./not_mutated_code', noun_mutated_list)
    print(len(mutated_list))
    print(len(noun_mutated_list))
    count = 0
    # mutated_tmp = random.sample(mutated_list, 12881)
    mutated_samples = []
    for item in tqdm.tqdm(mutated_list):
        fname = item.split('/')[-1]
        mutated_samples.append(fname)
    # print(mutated_samples)

    cname = []
    func = []
    label = []
    fp2 = open('true_mutated.txt', 'w')
    bar = tqdm.tqdm(total_path_list)
    with open('switch_trans.txt', 'w') as fp:
        for path in bar:
            flag = False
            fname = path.split('/')[-1]
            fdata = open(path, 'r').read().strip()
            fdata = get_data(fdata)
            # with open('./valid_code1/'+fname, 'w') as fp1:
            #     for line in fdata:
            #         if re.sub(r'(\s)+', '', line) != '': 
            #             fp1.write(line + '\n')
            if fname in mutated_samples:
                fdata1 = open('./mutated_code/' + fname, 'r').read().strip()
                fdata1 = get_data(fdata1)

                if fdata != fdata1:
                    fp2.write(fname + '\n')
                    count += 1
                    flag = True
            cname.append(fname)
            if flag == True and count <= 10000:
                fp.write(fname + '  1\n')
                fcode = open('./mutated_code/'+fname, 'r').read().strip()
                label.append('1')
            else:
                fp.write(fname + '  0\n')
                fcode = open(path, 'r').read().strip()
                label.append('0')
            func.append(fcode)
            bar.update()
    bar.close()
    fp2.close()

    print(count)
    df = pd.DataFrame({'id': cname, 'func': func, 'label': label})
    df.to_pickle('bug_detection.pkl')