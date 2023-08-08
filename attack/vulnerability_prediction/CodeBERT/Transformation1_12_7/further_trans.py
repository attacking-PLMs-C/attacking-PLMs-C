import os
import json
import tqdm
import argparse
import subprocess

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
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=int)
    args = parser.parse_args()

    file_list = []
    # get_file_path('../initial_code', file_path_list)
    with open('true_mutated1.txt', 'r') as fp:
        all_lines = fp.read().strip().split('\n')
        for line in tqdm.tqdm(all_lines):
            if line == '':
                continue
            file_list.append(line)

    bar = tqdm.tqdm(total=len(file_list))
    for i, path in  enumerate(file_list):
        fname = path
        # fid = fname.split('.')[0]
        path = './mutated_code1/' + fname
        # if i % 2 == 1:
        #     subprocess.check_output('cp ' + path + ' ./not_mutated_code', shell=True)
        # elif i % 2 == 0:
        subprocess.check_output('./runner2.sh ' + path + ' ' + fname + ' ' + str(args.action), shell=True)
        # print(path)
        # break
        bar.update()
    bar.close()