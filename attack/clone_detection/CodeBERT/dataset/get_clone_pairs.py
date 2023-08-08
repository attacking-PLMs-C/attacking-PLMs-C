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

#------------------------------start 
# 生成programs对应的pkl文件
def gen_programs_pkl(file_list):
    '''
    将所有的数据生成一个pkl 文件
    '''
    ids=[]
    files=[]
    labels=[]
    for file_path in file_list:
        paths=file_path.split('/')
        dirname=int(paths[-2])
        filename=int(paths[-1][:-2])
        id=dirname*10000+filename
        with open(file_path) as fp:
            file_code=fp.read()
        ids.append(id)
        files.append(file_code)
        labels.append(dirname)
    data={"index":ids,'code':files,'numclass':labels}
    df=pd.DataFrame(data)
    df.to_pickle('dataset/programs.pkl')
#-----------------------------end

#-----------------------------start
#生成数据对
def gen_file_pairs_subprocess(data):
    root_path='Program/'
    dirs,codes,dir_codes=data
    id1,code_list1,id2,code_list2,label=[],[],[],[],[]
    neg_dirs=list(dir_codes.keys())

    for i,code1 in enumerate(codes[:len(codes)]):
        for code2 in codes[i:]:
            id1.append(root_path+str(dirs)+'/'+str(code1)+'.c')
            s_code1 = open(root_path+str(dirs)+'/'+str(code1)+'.c').read().strip()
            # print(s_code1)
            # print(type(s_code1))
            code_list1.append(s_code1)
            id2.append(root_path+str(dirs)+'/'+str(code2)+'.c')
            s_code2 = open(root_path+str(dirs)+'/'+str(code2)+'.c').read().strip()
            code_list2.append(s_code2)
            label.append(1)
            neg_dir=random.choice(neg_dirs)
            while(neg_dir==dirs):
                neg_dir=random.choice(neg_dirs)
            neg_codes=dir_codes[neg_dir]
            neg_code=random.choice(neg_codes)
            code=random.choice([code1,code2])
            id1.append(root_path+str(dirs)+'/'+str(code)+'.c')
            s_code = open(root_path+str(dirs)+'/'+str(code)+'.c').read().strip()
            code_list1.append(s_code)
            id2.append(root_path+str(neg_dir)+'/'+str(neg_code)+'.c')
            s_neg_code = open(root_path+str(neg_dir)+'/'+str(neg_code)+'.c').read().strip()
            code_list2.append(s_neg_code)
            label.append(0)
    return id1,code_list1,id2,code_list2,label

def gen_file_pairs(file_list):
    '''
    生成对应的克隆对和非克隆对
    '''
    dir_codes={}
    for file_path in file_list: 
        # print(file_path)
        paths=file_path.split('/')
        dirname=int(paths[-2])
        filename=int(paths[-1][:-2])
        codes=dir_codes.get(dirname,list())
        codes.append(filename)
        dir_codes[dirname]=codes

    clone_count=0
    id1=[]
    id2=[]
    label=[]
    code1 = []
    code2 = []
    neg_dirs=list(dir_codes.keys())
    process=[]
    for dirs,codes in dir_codes.items():
        process.append([dirs,codes,dir_codes])
    with Pool(40) as p:
        for i in p.imap(gen_file_pairs_subprocess,tqdm.tqdm(process)):
            id1.extend(i[0])
            code1.extend(i[1])
            id2.extend(i[2])
            code2.extend(i[3])
            label.extend(i[4])
    data={
        "id1":id1,
        "code1":code1,
        "id2":id2,
        "code2":code2,
        "label":label
    }
    df=pd.DataFrame(data)
    # sampler=np.random.permutation(len(label))
    # df=df.take(sampler)
    # df.reset_index(inplace=True,drop=True)
    # df_test=df.sample(frac=0.001,replace=False)
    # # df_test.to_pickle("moj_clone_ids_001.pkl")
    # df_test=df.sample(frac=0.1,replace=False)
    # # df_test.to_pickle("moj_clone_ids_1.pkl")
    # df_test=df.sample(frac=0.01,replace=False)
    df.to_pickle("oj_clone_ids.pkl")
    # df.to_pickle("moj_clone_ids.pkl")

def split_pairs(filename):
    pairs=pd.read_pickle(filename)
    print(len(pairs))
    with open('data.jsonl',' w') as f:
        train_num=int(len(pairs)*0.8)
        val_num=int(len(pairs)*0.1)
        data1 = []
        data0 = []
        df_train=pairs.iloc[:train_num]
        for i, item in tqdm.tqdm(df_train.iterrows()):
            if item.label == 0:
                data0.append({'id1':item.id1,'code1':item.code1,'id2':item.id2,'code2':item.code2,'label':item.label})
            elif item.label == 1:
                data1.append({'id1':item.id1,'code1':item.code1,'id2':item.id2,'code2':item.code2,'label':item.label})
        with open('train_sampled.json', 'w') as fp:
            data = random.sample(data1, 45000) + random.sample(data0, 45000)
            random.shuffle(data)
            print(len(data))
            for item in data:
                js1 = {}
                js1['idx'] = item['id1'].split('/')[1]+'_'+item['id1'].split('/')[-1]+'.c'
                js1['func'] = item['code1']
                f.write(json.dumps(js1)+'\n')
                js2 = {}
                js2['idx'] = item['id2'].split('/')[1]+'_'+item['id2'].split('/')[-1]+'.c'
                js2['func'] = item['code2']
                f.write(json.dumps(js2)+'\n')
            json.dump(data, fp)

        data1 = []
        data0 = []
        df_val=pairs.iloc[train_num:train_num+val_num]
        for i, item in tqdm.tqdm(df_val.iterrows()):
            if item.label == 0:
                data0.append({'id1':item.id1,'code1':item.code1,'id2':item.id2,'code2':item.code2,'label':item.label})
            elif item.label == 1:
                data1.append({'id1':item.id1,'code1':item.code1,'id2':item.id2,'code2':item.code2,'label':item.label})
        with open('val_sampled.json', 'w') as fp:
            data = random.sample(data1, 2000) + random.sample(data0, 2000)
            random.shuffle(data)
            print(len(data))
            for item in data:
                js1 = {}
                js1['idx'] = item['id1'].split('/')[1]+'_'+item['id1'].split('/')[-1]+'.c'
                js1['func'] = item['code1']
                f.write(json.dumps(js1)+'\n')
                js2 = {}
                js2['idx'] = item['id2'].split('/')[1]+'_'+item['id2'].split('/')[-1]+'.c'
                js2['func'] = item['code2']
                f.write(json.dumps(js2)+'\n')
            json.dump(data, fp)

        data1 = []
        data0 = []
        df_test=pairs.iloc[train_num+val_num:]
        for i, item in tqdm.tqdm(df_test.iterrows()):
            if item.label == 0:
                data0.append({'id1':item.id1,'code1':item.code1,'id2':item.id2,'code2':item.code2,'label':item.label})
            elif item.label == 1:
                data1.append({'id1':item.id1,'code1':item.code1,'id2':item.id2,'code2':item.code2,'label':item.label})
        with open('test_sampled.json', 'w') as fp:
            data = random.sample(data1, 2000) + random.sample(data0, 2000)
            random.shuffle(data)
            print(len(data))
            for item in data:
                js1 = {}
                js1['idx'] = item['id1'].split('/')[1]+'_'+item['id1'].split('/')[-1]+'.c'
                js1['func'] = item['code1']
                f.write(json.dumps(js1)+'\n')
                js2 = {}
                js2['idx'] = item['id2'].split('/')[1]+'_'+item['id2'].split('/')[-1]+'.c'
                js2['func'] = item['code2']
                f.write(json.dumps(js2)+'\n')
            json.dump(data, fp)

    # df_train.reset_index(inplace=True,drop=True)
    # df_val.reset_index(inplace=True,drop=True)
    # df_test.reset_index(inplace=True,drop=True)
    # df_train.to_pickle("dataset/oj_clone_ids_train.pkl")
    # df_val.to_pickle("dataset/oj_clone_ids_val.pkl")
    # df_test.to_pickle("dataset/oj_clone_ids_test.pkl")
    
#-----------------------------end 

def get_data_json():
    code_labels = open('../code_labels.txt', 'w')
    with open('data.jsonl', 'w') as f:
        data = json.load(open('train_sampled.json', 'r'))
        for item in data:
            js1 = {}
            js1['idx'] = item['id1'].split('/')[1]+'_'+item['id1'].split('/')[-1]
            js1['func'] = item['code1']
            f.write(json.dumps(js1)+'\n')
            js2 = {}
            js2['idx'] = item['id2'].split('/')[1]+'_'+item['id2'].split('/')[-1]
            js2['func'] = item['code2']
            f.write(json.dumps(js2)+'\n')
            code_labels.write(js1['idx']+'  '+js2['idx']+'  '+str(item['label'])+'\n')
        
        data = json.load(open('val_sampled.json', 'r'))
        for item in data:
            js1 = {}
            js1['idx'] = item['id1'].split('/')[1]+'_'+item['id1'].split('/')[-1]
            js1['func'] = item['code1']
            f.write(json.dumps(js1)+'\n')
            js2 = {}
            js2['idx'] = item['id2'].split('/')[1]+'_'+item['id2'].split('/')[-1]
            js2['func'] = item['code2']
            f.write(json.dumps(js2)+'\n')
            code_labels.write(js1['idx']+'  '+js2['idx']+'  '+str(item['label'])+'\n')

        data = json.load(open('test_sampled.json', 'r'))
        for item in data:
            js1 = {}
            js1['idx'] = item['id1'].split('/')[1]+'_'+item['id1'].split('/')[-1]
            js1['func'] = item['code1']
            f.write(json.dumps(js1)+'\n')
            js2 = {}
            js2['idx'] = item['id2'].split('/')[1]+'_'+item['id2'].split('/')[-1]
            js2['func'] = item['code2']
            f.write(json.dumps(js2)+'\n')
            code_labels.write(js1['idx']+'  '+js2['idx']+'  '+str(item['label'])+'\n')
    code_labels.close()

if __name__=='__main__':
    # root_path="Program/" 
    # # step1
    # # root_path=args.dirname
    # # get_different_test(root_path,2000,"oj_clone_ids_train_mixall.pkl")
    # # 获取所有文件名 
    # file_list=[]
    # dir_list=[]
    # get_file_path(root_path,file_list,dir_list)
    # file_list.sort() 
    # filter_file(file_list)
    # print(file_list)
    
    # # # # 从单个文件构造成整体的pickle
    # # print('[+] gen programs.pkl...')
    # # gen_programs_pkl(file_list)
    
    # # # 生成整体的克隆文件对
    # print('[+] gen clone pairs...')
    # gen_file_pairs(file_list)
    
    # # # 划分出训练集验证集训练集
    # print('[+] split clone pairs...')
    # split_pairs("oj_clone_ids.pkl")

    # get_data_json()
    pairs=pd.read_pickle('oj_clone_ids.pkl')
    print(len(pairs))

    
 
 