import os
import sys
import torch
from collections import Counter
import tqdm
import pickle as pkl
import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer


def get_uc_filtering(code):
    '''过滤掉中文字符'''
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    code = re.sub(pattern, '', code)
    return code

def code_filtering(s_code):
    '''去除code中的注释、中文'''
    source_code = s_code
    source_code = re.sub(r'/\*([\s\S]*?)\*/', '', source_code)
    code = get_uc_filtering(source_code)
    return code

def get_feature_token(data_path, tokenizer, store_path):
    vec_pkl = []
    with open(data_path) as fp:
        all_data = json.load(fp)
        for data in tqdm.tqdm(all_data):
            id_ = data['id1'] + '_' + data['id2']
            # with open('./valid_code/'+id_, 'w') as fp:
            #     fp.write(fcode.strip())
            label = data['label']
            code1 = data['code1']
            code1 = re.sub(r'(\s)+', ' ', code1).strip()
            code1 = code_filtering(code1)
            ftokens1 = ''.join(code1.split(' '))
            ftokens1 = tokenizer.tokenize(ftokens1)[:510]
            ftokens1 = [tokenizer.cls_token] + ftokens1 + [tokenizer.sep_token]
            # fids1 = tokenizer.convert_tokens_to_ids(ftokens1)
            code2 = data['code2']
            code2 = re.sub(r'(\s)+', ' ', code2).strip()
            code2 = code_filtering(code2)
            ftokens2 = ''.join(code2.split(' '))
            ftokens2 = tokenizer.tokenize(ftokens2)[:510]
            ftokens2 = [tokenizer.cls_token] + ftokens2 + [tokenizer.sep_token]
            # fids2 = tokenizer.convert_tokens_to_ids(ftokens2)
            ftokens = ftokens1+ftokens2
            if len(ftokens) > 512:
                ftokens = ftokens[:511] + [tokenizer.sep_token]
            fids = tokenizer.convert_tokens_to_ids(ftokens)
            padding_length = 512 - len(fids)
            fids += [tokenizer.pad_token_id] * padding_length
            vec_pkl.append([id_, fids, label])
    # data_path = data_path.replace('.pkl', '_token.pkl')
    pkl.dump(vec_pkl, open(store_path, 'wb'))

if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    print('[+] transform tokens to vectors...')
    train_data_path = './dataset/train_sampled.json'
    val_data_path = './dataset/val_sampled.json'
    test_data_path = './dataset/test_sampled.json'
    get_feature_token(train_data_path, tokenizer, './data/train_set_token.pkl')
    get_feature_token(val_data_path, tokenizer, './data/val_set_token.pkl')
    get_feature_token(test_data_path, tokenizer, './data/test_set_token.pkl')
    # get_feature_token('./all_set.pkl', tokenizer)