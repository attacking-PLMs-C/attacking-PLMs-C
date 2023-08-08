import pickle
import os
import re
import pandas as pd
import json
import tqdm

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

def get_uc_filtering(code):
    '''过滤掉中文字符'''
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    code = re.sub(pattern, '', code)
    return code

def code_filtering(file_list):
    '''去除code中的注释、中文'''
    count = 0  # 统计有问题代码数量
    new_file_list = []
    pbar = tqdm.tqdm(file_list)
    pbar.set_description('comments filtering')
    for file in pbar:
        '''注意windows目录下的文件路径格式'''
        file_name = file.split('/')[-1]
        # print(file_name)
        with open(file, 'r') as fp:
            new_file = './filtered_code/' + file_name
            new_file_list.append(new_file)
            # if file_name == '79a54f30c8ba02cbf2b02c650120246b260977ec_19424.c':
            #     continue
            source_code = fp.read()
            source_code = re.sub(r'/\*([\s\S]*?)\*/', '', source_code)
            source_code = source_code.split('\n')
            with open(new_file, 'w') as fp1:
                for code_line in source_code:
                    code_line = re.sub(r'//[\s\S]*', '', code_line)
                    fp1.write(code_line + '\n')
            fp1.close()
            pbar.update()
    pbar.close()

    bar = tqdm.tqdm(new_file_list)
    bar.set_description('unChinese filtering')
    for file in bar:
        try:
            with open(file, 'rb') as fp:
                code = fp.read()
                code = code.decode('utf-8', 'ignore')
                code = get_uc_filtering(code)
        except Exception as e:
            count += 1
            print(file)
        bar.update()
    bar.close()
    print(count)
    return new_file_list

if __name__ == "__main__":
    # root = "./function.json"
    # code_path = "./code/"
    # scode = json.load(open(root, 'r'))
    # print(len(scode))
    # bar = tqdm.tqdm(enumerate(scode))
    # with open('code_labels.txt', 'w') as fp:
    #     for i, item in bar:
    #         with open(code_path+item["commit_id"]+"_"+str(i).zfill(5)+".c", 'w') as co:
    #             co.write(item["func"])
    #         fp.write(item["commit_id"] + "_" + str(i).zfill(5) + ".c" + "  "+ str(item["target"]) + '\n')
    #         bar.update()
    #     bar.close()

    code_file_list = []
    '''获取文件路径'''
    get_file_path('./code', code_file_list)
    '''去除源代码中的注释和中文'''
    # code_file_list = ['test.c', 'test1.c', 'test2.c', 'test3.c']
    print('[+] code filtering...')
    code_filtering(code_file_list)