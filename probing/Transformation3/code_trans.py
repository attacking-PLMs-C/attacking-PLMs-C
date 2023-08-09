import os
import json
import tqdm
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
    file_path_list = []
    get_file_path('./valid_code', file_path_list)
    bar = tqdm.tqdm(total=len(file_path_list))
    for i, path in  enumerate(file_path_list):
        fname = path.split('/')[-1]
        # if i % 2 == 1:
        #     subprocess.check_output('cp ' + path + ' ./not_mutated_code', shell=True)
        # elif i % 2 == 0:
        subprocess.check_output('./rename.sh ' + path + ' ' + fname + ' 3', shell=True)
        # print(path)
        # break
        bar.update()
    bar.close()