import re
import os
from collections import Counter
import tqdm
import operator
import random
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=int)
    args = parser.parse_args()

    mutated_list = []
    # noun_mutated_list = []
    get_file_path('./mutated_code'+str(args.action), mutated_list)
    # get_file_path('./not_mutated_code', noun_mutated_list)
    print(len(mutated_list))
    # print(len(noun_mutated_list))
    count = 0

    bar = tqdm.tqdm(mutated_list)
    with open('./true_mutated' + str(args.action) + '.txt', 'w') as fp:
        for path in bar:
            fname = path.split('/')[-1]
            # fid = fname.split('.')[0].split('_')[-1]
            source_path = './mutated_code1/' + fname
            f_mutated = open(path,'r').read().strip()
            f_mutated = get_data(f_mutated)
            f_source = open(source_path, 'r').read().strip()
            f_source = get_data(f_source)
            if f_mutated != f_source:
                # print(fname)
                fp.write(path.split('/')[-1] + '\n')
            # break
            bar.update()
        bar.close()