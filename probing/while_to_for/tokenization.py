import os
import sys
from collections import Counter
import tqdm
import pickle as pkl
import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer


def get_data_split(pkl_path):
    db = pd.read_pickle(pkl_path)
    train_data, test_data = train_test_split(db, test_size=0.2, random_state=9, stratify=db['label'])
    # print(val_test_data)
    store_path = './data'
    if not os.path.exists(store_path):
        os.mkdir(store_path)

    train_set_path = store_path + '/train_set.pkl'
    test_set_path = store_path + '/test_set.pkl'
    # train_set = pd.DataFrame()
    train_data.to_pickle(train_set_path)
    test_data.to_pickle(test_set_path)

def get_feature_token(data_path, tokenizer):
    data = pd.read_pickle(data_path)
    vec_pkl = []
    bar = tqdm.tqdm(total=len(data))
    for _, item in data.iterrows():
        id = item.id
        fcode = item.func
        label = item.label
        fcode = re.sub(r'(\s)+', ' ', fcode).strip()
        ftokens = ''.join(fcode.split(' '))
        ftokens = tokenizer.tokenize(ftokens)[:510]
        ftokens = [tokenizer.cls_token] + ftokens + [tokenizer.sep_token]
        fids = tokenizer.convert_tokens_to_ids(ftokens)
        padding_length = 512 - len(fids)
        fids += [tokenizer.pad_token_id] * padding_length
        vec_pkl.append([id, fids, label])
        bar.update()
    bar.close()
    data_path = data_path.replace('.pkl', '_token.pkl')
    pkl.dump(vec_pkl, open(data_path, 'wb'))

if __name__ == "__main__":
    '''划分数据集'''
    print('[+] split dataset...')
    get_data_split('./bug_detection.pkl')

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    print('[+] transform tokens to vectors...')
    train_data_path = './data/train_set.pkl'
    test_data_path = './data/test_set.pkl'
    get_feature_token(train_data_path, tokenizer)
    get_feature_token(test_data_path, tokenizer)