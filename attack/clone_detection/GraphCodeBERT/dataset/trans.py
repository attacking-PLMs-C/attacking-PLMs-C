import os
import re
import sys 
import tqdm
import json
import random
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from multiprocessing import Pool
# from utils import get_code_token,SPECIAL_WORD
sys.path.append('.')

'''
这个文件主要完成
1. poj 文件的过滤
2. 从poj 文件生成整体的programs.pkl 文件
3. 生成clone 和 非 clone 相对应
'''

def get_file_path(root_path,file_list,dir_list):
    '''获取所有的文件'''
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path,dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            get_file_path(dir_file_path,file_list,dir_list)
        else:
            file_list.append(dir_file_path)

#------------------------------start 
# 文件过滤
def find_unchinese(file):     
    pattern = re.compile(r'[\u4e00-\u9fa5]')     
    unchinese = re.sub(pattern,"",file) 
    return unchinese

def filter_file(file_list):
    '''
    运行的时候先运行这个函数过滤掉有问题的代码
    过滤掉有问题的代码，有问题的代码有281个'''
    count=0
    for file_path in file_list:
        try:
            with open(file_path,'rb') as fp:
                data=fp.read()
                data=data.decode('utf-8')
                data=find_unchinese(data)
        except Exception as e:
            count=count+1
            print(file_path) 
            os.remove(file_path)
            continue
        with open(file_path,'w') as fp:
            fp.write(data)
    # print(count)
#------------------------------end

if __name__=='__main__':
    root_path="ProgramData/" 
    # step1
    # root_path=args.dirname
    # get_different_test(root_path,2000,"oj_clone_ids_train_mixall.pkl")
    # 获取所有文件名 
    file_list=[]
    dir_list=[]
    get_file_path(root_path,file_list,dir_list)
    filter_file(file_list)
    
    for file_path in tqdm.tqdm(file_list):
        paths = file_path.split('/')
        if not os.path.exists('Program/'+paths[1]):
            os.mkdir('Program/'+paths[1])
        t_path = 'Program/' + paths[1] + '/' + paths[-1].split('.')[0] + '.c'
        # print(t_path)
        with open(t_path, 'w') as fp:
            s_code = open(file_path, 'r').read().strip()
            fp.write(s_code)
        # break

    
 
 