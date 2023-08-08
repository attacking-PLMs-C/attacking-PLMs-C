import os
import sys
import torch
from collections import Counter
import tqdm
import argparse
import pickle as pkl
import json
import re
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer

def get_tokenization(target_code1, target_code2, tokenizer):
    target_code1 = re.sub(r'(\s)+', ' ', target_code1).strip()
    target_tokens1 = ''.join(target_code1.split(' '))
    target_tokens1 = tokenizer.tokenize(target_tokens1)[:510]
    target_tokens1 = [tokenizer.cls_token] + target_tokens1 + [tokenizer.sep_token]
    target_code2 = re.sub(r'(\s)+', ' ', target_code2).strip()
    target_tokens2 = ''.join(target_code2.split(' '))
    target_tokens2 = tokenizer.tokenize(target_tokens2)[:510]
    target_tokens2 = [tokenizer.cls_token] + target_tokens2 + [tokenizer.sep_token]
    target_tokens = target_code1 + target_code2
    if len(target_tokens) > 512:
        target_tokens = target_tokens[:511] + [tokenizer.sep_token]
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    padding_length = 512 - len(target_ids)
    target_ids += [tokenizer.pad_token_id] * padding_length

    return target_ids

def get_feature_vector(fcode1, fcode2, tokenizer):
    fcode1 = re.sub(r'(\s)+', ' ', fcode1).strip()
    ftokens1 = ''.join(fcode1.split(' '))
    ftokens1 = tokenizer.tokenize(ftokens1)[:510]
    ftokens1 = [tokenizer.cls_token] + ftokens1 + [tokenizer.sep_token]
    fcode2 = re.sub(r'(\s)+', ' ', fcode2).strip()
    ftokens2 = ''.join(fcode2.split(' '))
    ftokens2 = tokenizer.tokenize(ftokens2)[:510]
    ftokens2 = [tokenizer.cls_token] + ftokens2 + [tokenizer.sep_token]
    ftokens = ftokens1 + ftokens2
    if len(ftokens) > 512:
        ftokens = ftokens[:511] + [tokenizer.sep_token]
    fids = tokenizer.convert_tokens_to_ids(ftokens)
    padding_length = 512 - len(fids)
    fids += [tokenizer.pad_token_id] * padding_length
    return fids

def get_prediction(f_code1, fcode2, tokenizer, model, config):
    f_ids = get_feature_vector(f_code1, fcode2, tokenizer)
    f_ids = torch.tensor(f_ids).to(config.device)
    out, hidden_out, atten_out = model(f_ids)
    predict = torch.max(out.data, 1)[1].item()
    return predict, hidden_out, atten_out

def get_file_list(root_path, file_list):
    with open(root_path, 'r') as fp:
        all_lines = fp.read().strip().split('\n')
        for line in all_lines:
            file_list.append(line)
    return file_list

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
