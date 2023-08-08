import os
import json
import tqdm
import re
from collections import Counter
import operator
import random
import argparse
import pandas as pd
import subprocess
import itertools
from multiprocessing import Pool

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

if __name__ == '__main__':
    file_path_list = get_file_path('./RM', [])
    # print(file_path_list)
    for path in tqdm.tqdm(file_path_list):
        if path.endswith('.c'):
            subprocess.check_output('rm -rf '+path, shell=True)