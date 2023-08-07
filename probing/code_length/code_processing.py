import os 
import pickle as  pkl
import tqdm
import pandas as pd

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
    id  = []
    func = []
    label = []
    id_to_label = {}

    '''提取函数名和label信息'''
    with open('code_length.txt', 'r') as fp:
        msg = fp.read().split('\n')
        # print(msg)
        for line in msg:
            fname = line.split('  ')[0]
            flabel = line.split('  ')[-1]
            id_to_label[fname] = flabel

    # print(id_to_label)
    bar = tqdm.tqdm(file_path_list)
    for path in bar:
        fname = path.split('/')[-1]
        if fname not in id_to_label:
            continue
        with open(path, 'r') as fp:
            '''去除代码结尾多余的空行'''
            fcode = fp.read().strip()
            func.append(fcode)
        id.append(fname)
        label.append(id_to_label[fname])
        bar.update()
    bar.close()

    '''生成整体pkl文件'''
    print('[+] generate bug_detection.pkl...')
    func_pkl = {'id': id, 'func': func, 'label': label}
    df = pd.DataFrame(func_pkl)
    df.to_pickle('bug_detection.pkl')