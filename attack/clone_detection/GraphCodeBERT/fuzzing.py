import os
import sys
import torch
from collections import Counter
import tqdm
import argparse
import pickle as pkl
import json
from torchsummary import summary 
import re
import multiprocessing
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer
from utils import *
import pandas as pd
import warnings
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from importlib import import_module
import operator
from Encoder import Model
from torch.nn import DataParallel
sys.path.append('.')
import glob
import logging
import pickle
import shutil
sys.path.append('.')
sys.path.append('./python_parser')
from run_parser import extract_dataflow
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from Encoder import Model
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

surface_mutants = {'0': '1'}
syntax_mutants = {'1': '1_12', '2': '12'}
# syntax_mutants = {'1': '2', '2': '7', '4': '1_2', '5': '1_7', '7': '1_2_7', '11': '2_7'}
semantic_mutants = {'3': '2', '4': '3', '5': '7', '6': '1_2', '7': '1_3', '8': '1_7', '9': '1_2_3', '10': '1_2_7', '11': '1_3_7', '12': '1_2_3_7', '13': '12_2', '14': '12_3', '15': '12_7',
                      '16': '12_2_3', '17': '12_2_7', '18': '12_3_7', '19': '12_2_3_7', '20': '1_12_2', '21': '1_12_3', '22': '1_12_7', '23': '1_12_2_3', '24': '1_12_2_7', '25': '1_12_3_7',
                      '26': '1_12_2_3_7', '27': '2_3', '28': '2_7', '29': '3_7', '30': '2_3_7'}
mutation_index = {'0': '1', '1': '1_12', '2': '12', '3': '2', '4': '3', '5': '7', '6': '1_2', '7': '1_3', '8': '1_7', '9': '1_2_3', '10': '1_2_7', '11': '1_3_7', '12': '1_2_3_7', '13': '12_2', '14': '12_3', '15': '12_7',
                      '16': '12_2_3', '17': '12_2_7', '18': '12_3_7', '19': '12_2_3_7', '20': '1_12_2', '21': '1_12_3', '22': '1_12_7', '23': '1_12_2_3', '24': '1_12_2_7', '25': '1_12_3_7',
                      '26': '1_12_2_3_7', '27': '2_3', '28': '2_7', '29': '3_7', '30': '2_3_7'}

def get_candidates(args, initial_seeds, tokenizer, model, config, initial_candidates, indices, file_to_label):
    model.eval()
    with torch.no_grad():
        for fn, item in tqdm.tqdm(initial_seeds.items()):
            sub_features = {}
            s_name = fn.split('__')[0]
            s_path = './initial_code/' + s_name
            s_code = open(s_path, 'r').read().strip()
            for sub_seed in item:
                s_path2 = './code2/' + sub_seed[2]
                s_code2 = open(s_path2, 'r').read().strip()
                s_data = TextDataset(tokenizer, args, s_code, s_code2, fn, int(file_to_label[fn]))
                s_dataloader = DataLoader(s_data, batch_size=1)
                for batch in s_dataloader:
                    input_ids = batch[0].to(args.device)
                    atten_mask = batch[1].to(args.device)
                    position_idx = batch[2].to(args.device)
                    label = batch[3].to(args.device)
                    idx = batch[4]
                    _, _, s_hiddenout, s_attentionout = model(input_ids, atten_mask, position_idx, label)

                angle_out = torch.index_select(s_hiddenout, dim=1, index=indices)[:,:,0,:]
                # E_out = torch.index_select(s_attentionout, dim=1, index=indices-1)
                f_path = './mutated_candidates/mutated_code'+sub_seed[0]+'/'+sub_seed[1]
                f_code = open(f_path, 'r').read().strip()
                f_data = TextDataset(tokenizer, args, f_code, s_code2, fn, int(file_to_label[fn]))
                f_dataloader = DataLoader(f_data, batch_size=1)
                for batch in f_dataloader:
                    input_ids = batch[0].to(args.device)
                    atten_mask = batch[1].to(args.device)
                    position_idx = batch[2].to(args.device)
                    label = batch[3].to(args.device)
                    idx = batch[4]
                    _, logit, f_hiddenout, f_attentionout = model(input_ids, atten_mask, position_idx, label)
                    f_predict = int(logit.cpu().numpy()>0.5)

                angle_out1 = torch.index_select(f_hiddenout, dim=1, index=indices)[:,:,0,:]
                # E_out1 = torch.index_select(f_attentionout, dim=1, index=indices-1)
                metric = F.cosine_similarity(angle_out, angle_out1, dim=-1)
                metric = torch.acos(metric) * 180 / 3.1415926
                metric = torch.mean(metric)
                if metric <= 0.0:
                    continue
                feat1 = [f_path, metric.item(), f_predict]
                if fn not in sub_features:
                    sub_features[fn] = [feat1]
                else:
                    sub_features[fn].append(feat1)
            
            for pi, sf in sub_features.items():
                initial_candidates[pi] = sorted(sf, key=operator.itemgetter(1), reverse=True)[:20]
    return initial_candidates

def get_initial_seeds(file, t_file, k, v, initial_seeds):
    if file in v and file not in initial_seeds:
        initial_seeds[file] = [k]
    elif t_file in v and t_file not in initial_seeds:
        initial_seeds[t_file] = [k]
    elif file in v and file in initial_seeds:
        initial_seeds[file].append(k)
    elif t_file in v and t_file in initial_seeds:
        initial_seeds[t_file].append(k)
    return initial_seeds

def fit_function(args, source_path, tmp_path, target_path, path2, cpair, file_to_label, tokenizer, model, config, linguistic_indices):
    code2 = open(path2, 'r').read().strip()
    s_code = open(source_path, 'r').read().strip()
    s_data = TextDataset(tokenizer, args, s_code, code2, cpair, int(file_to_label[cpair]))
    s_dataloader = DataLoader(s_data, batch_size=1)
    for batch in s_dataloader:
        input_ids = batch[0].to(args.device)
        atten_mask = batch[1].to(args.device)
        position_idx = batch[2].to(args.device)
        label = batch[3].to(args.device)
        # idx = batch[4]
        _, _, s_hiddenout, s_attentionout = model(input_ids, atten_mask, position_idx, label)

    angle_out = torch.index_select(s_hiddenout, dim=1, index=linguistic_indices)[:,:,0,:]
    E_out = torch.index_select(s_attentionout, dim=1, index=linguistic_indices-1)

    tmp_code = open(tmp_path, 'r').read().strip()
    tmp_data = TextDataset(tokenizer, args, tmp_code, code2, cpair, int(file_to_label[cpair]))
    tmp_dataloader = DataLoader(tmp_data, batch_size=1)
    for batch in tmp_dataloader:
        input_ids = batch[0].to(args.device)
        atten_mask = batch[1].to(args.device)
        position_idx = batch[2].to(args.device)
        label = batch[3].to(args.device)
        # idx = batch[4]
        _, _, tmp_hiddenout, tmp_attentionout = model(input_ids, atten_mask, position_idx, label)
    tmp_angle_out = torch.index_select(tmp_hiddenout, dim=1, index=linguistic_indices)[:,:,0,:]
    tmp_E_out = torch.index_select(tmp_attentionout, dim=1, index=linguistic_indices-1)

    t_code = open(target_path, 'r').read().strip()
    t_data = TextDataset(tokenizer, args, t_code, code2, cpair, int(file_to_label[cpair]))
    t_dataloader = DataLoader(t_data, batch_size=1)
    for batch in t_dataloader:
        input_ids = batch[0].to(args.device)
        atten_mask = batch[1].to(args.device)
        position_idx = batch[2].to(args.device)
        label = batch[3].to(args.device)
        # idx = batch[4]
        _, _, t_hiddenout, t_attentionout = model(input_ids, atten_mask, position_idx, label)
    t_angle_out = torch.index_select(t_hiddenout, dim=1, index=linguistic_indices)[:,:,0,:]
    t_E_out = torch.index_select(t_attentionout, dim=1, index=linguistic_indices-1)

    cs_value = F.cosine_similarity(angle_out, tmp_angle_out, dim=-1)
    adho = torch.acos(cs_value) * 180 / 3.1415926
    adho = torch.mean(adho)
    edao = torch.norm(E_out-tmp_E_out, p=2)

    cs_value1 = F.cosine_similarity(tmp_angle_out, t_angle_out, dim=-1)
    adho1 = torch.acos(cs_value1) * 180 / 3.1415926
    adho1 = torch.mean(adho1)
    edao1 = torch.norm(tmp_E_out-t_E_out, p=2)

    if adho1 >= adho and edao1 >= edao:
        return True
    else:
        return False

def fuzzer(config, args, model, tokenizer, file_to_pairs, file_to_label, mutated_files_list):
    model.eval()
    surface_files = []
    code_rename = []
    tmp = []
    for_to_while = []
    switch = []
    code_redefine = []
    # true_mutated_files = {}
    surface_mutated_files = {}
    syntax_mutated_files = {}
    semantic_mutated_files = {}
    sub_query = 0.0

    surface_indices = torch.tensor([5]).to(args.device)
    syntax_indices = torch.tensor([5,6,7]).to(args.device)
    semantic_indices = torch.tensor([8,9,10]).to(args.device)
    get_file_list('./mutated_candidates/true_mutated1.txt', tmp)

    variable_candidates = {}
    surface_success = []
    syntax_sucess = []

    code_rename = []
    if os.path.exists('./surface_seeds.json'):
        surface_seeds = json.load(open('./surface_seeds.json'))
        syntax_seeds = json.load(open('./syntax_seeds.json'))
        semantic_seeds = json.load(open('./semantic_seeds.json'))
    else:
        with torch.no_grad():
            for t_file in tqdm.tqdm(tmp):
                t_name = t_file.split('-')[0] + '-' + t_file.split('-')[1] + '.c'
                if t_name not in variable_candidates:
                    variable_candidates[t_name] = [t_file]
                else:
                    variable_candidates[t_name].append(t_file)
                if t_name not in code_rename:
                    code_rename.append(t_name)
            # print(variable_candidates)
            # print(code_rename, len(code_rename))
            
            surface_files = json.load(open('./surface_files.json'))
            # file_to_candidates = {}
            # for file in tqdm.tqdm(code_rename):
            #     s_name = file.split('-')[0] + '.c'
            #     s_path = './initial_code/' + s_name
            #     s_code = open(s_path, 'r').read().strip()
            #     s_data = TextDataset(tokenizer, args, s_code, s_path, int(file_to_label[s_name]))
            #     # s_sampler = SequentialSampler(s_data)
            #     s_dataloader = DataLoader(s_data, batch_size=1)
            #     for batch in s_dataloader:
            #         input_ids = batch[0].to(args.device)
            #         atten_mask = batch[1].to(args.device)
            #         position_idx = batch[2].to(args.device)
            #         label = batch[3].to(args.device)
            #         idx = batch[4]
            #         _, logit, s_hiddenout, s_attentionout = model(input_ids, atten_mask, position_idx, label)
            #         lab = int(logit.cpu().numpy()>0.5)
            #     angle_out = torch.index_select(s_hiddenout, dim=1, index=surface_indices)[:,:,0,:]
            #     E_out = torch.index_select(s_attentionout, dim=1, index=surface_indices-1)
            #     max_distance = 0.0
            #     best_cand = ''
            #     tag1 = True
            #     for sub_cand in variable_candidates[file]:
            #         sub_path = './mutated_candidates/mutated_code1/' + sub_cand
            #         sub_code = open(sub_path, 'r').read().strip()
            #         sub_data = TextDataset(tokenizer, args, sub_code, sub_path, int(file_to_label[s_name]))
            #         sub_dataloader = DataLoader(sub_data, batch_size=1)
            #         for sub_batch in sub_dataloader:
            #             input_ids = sub_batch[0].to(args.device)
            #             atten_mask = sub_batch[1].to(args.device)
            #             position_idx = sub_batch[2].to(args.device)
            #             label = sub_batch[3].to(args.device)
            #             idx = sub_batch[4]
            #             _, logit, sub_hiddenout, sub_attentionout = model(input_ids, atten_mask, position_idx, label)
            #             pred = int(logit.cpu().numpy()>0.5)
            #         sub_angle_out = torch.index_select(sub_hiddenout, dim=1, index=surface_indices)[:,:,0,:]
            #         sub_E_out = torch.index_select(sub_attentionout, dim=1, index=surface_indices-1)
            #         if lab != pred:
            #             best_cand = sub_cand
            #             break
            #         if args.d_type == 'a_distance':
            #             metric = F.cosine_similarity(angle_out, sub_angle_out, dim=-1)
            #             metric = torch.acos(metric) * 180 / 3.1415926
            #             metric = torch.mean(metric)
            #         else:
            #             metric = torch.norm(E_out-sub_E_out, p=2)
            #         if metric <= 0.0:
            #             tag1 = False
            #             break
            #         if max_distance < metric:
            #             max_distance = metric
            #             best_cand = sub_cand
            #     if tag1 == False:
            #         continue
            #     if s_name not in file_to_candidates:
            #         file_to_candidates[s_name] = [best_cand]
            #     elif s_name in file_to_candidates:
            #         file_to_candidates[s_name].append(best_cand)
            # # with open('./file_to_candidates.json', 'w') as fp:
            # #     json.dump(file_to_candidates, fp)
            # # print(file_to_candidates)

            # for fname, cand_list in tqdm.tqdm(file_to_candidates.items()):
            #     for i in range(len(cand_list)):
            #         if cand_list[i] == '':
            #             continue
            #         sub_query += 1.0
            #         if i == len(cand_list) - 1:
            #             surface_files.append(cand_list[i])
            #             break
            #         f_path = './mutated_candidates/mutated_code1/' + cand_list[i]
            #         f_code = open(f_path, 'r').read().strip()
            #         f_data = TextDataset(tokenizer, args, f_code, f_path, int(file_to_label[fname]))
            #         f_dataloader = DataLoader(f_data, batch_size=1)
            #         for f_batch in f_dataloader:
            #             input_ids = f_batch[0].to(args.device)
            #             atten_mask = f_batch[1].to(args.device)
            #             position_idx = f_batch[2].to(args.device)
            #             label = f_batch[3].to(args.device)
            #             idx = f_batch[4]
            #             _, logit, _, _ = model(input_ids, atten_mask, position_idx, label)
            #             predict = int(logit.cpu().numpy()>0.5)
            #         if predict != int(file_to_label[fname]):
            #             surface_files.append(cand_list[i])
            #             break
            
            '''通过ADHO和EDAO生成新种子'''
            surface_seeds = {}
            syntax_seeds = {}
            semantic_seeds = {}
            for file in tqdm.tqdm(code_rename):
                s_name = file.split('-')[0] + '.c'
                # s_path = './initial_code/' + s_name
                # s_code = open(s_path, 'r').read().strip()
                for item in file_to_pairs[s_name]:  #item: [id2, label]
                    adopted_seed = []
                    cpair = s_name+'__'+item[0]
                    for sub_file in mutated_files_list[s_name]:  #sub_file: [idx, mutated_file]
                        m_seed = sub_file[0].split('_')
                        if sub_file[0] in surface_mutants.values():
                            adopted_seed.append(sub_file[0])
                            surface_seeds[cpair] = [[sub_file[0], sub_file[1], item[0]]]
                        elif sub_file[0] in syntax_mutants.values():
                            if len(m_seed) == 1:
                                adopted_seed.append(sub_file[0])
                                if cpair not in syntax_seeds:
                                    syntax_seeds[cpair] = [[sub_file[0], sub_file[1], item[0]]]
                                else:
                                    syntax_seeds[cpair].append([sub_file[0], sub_file[1], item[0]])
                            elif len(m_seed) == 2:
                                if sub_file[0][:-(len(m_seed[-1])+1)] in adopted_seed:
                                    source_path = './initial_code/'+s_name
                                    tmp_path = './mutated_candidates/mutated_code'+sub_file[0][:-(len(m_seed[-1])+1)]+'/'+sub_file[1]
                                    target_path = './mutated_candidates/mutated_code'+sub_file[0]+'/'+sub_file[1]
                                    path2 = './code2/'+item[0]
                                    if fit_function(args, source_path, tmp_path, target_path, path2, cpair, file_to_label, tokenizer, model, config, semantic_indices):
                                        adopted_seed.append(sub_file[0])
                                        if cpair not in syntax_seeds:
                                            syntax_seeds[cpair] = [[sub_file[0], sub_file[1], item[0]]]
                                        else:
                                            syntax_seeds[cpair].append([sub_file[0], sub_file[1], item[0]])
                        elif sub_file[0] in semantic_mutants.values():
                            if len(m_seed) == 1:
                                adopted_seed.append(sub_file[0])
                                if cpair not in semantic_seeds:
                                    semantic_seeds[cpair] = [[sub_file[0], sub_file[1], item[0]]]
                                else:
                                    semantic_seeds[cpair].append([sub_file[0], sub_file[1], item[0]])
                            elif len(m_seed) == 2:
                                if sub_file[0][:-(len(m_seed[-1])+1)] in adopted_seed:
                                    source_path = './initial_code/'+s_name
                                    tmp_path = './mutated_candidates/mutated_code'+sub_file[0][:-(len(m_seed[-1])+1)]+'/'+sub_file[1]
                                    target_path = './mutated_candidates/mutated_code'+sub_file[0]+'/'+sub_file[1]
                                    path2 = './code2/'+item[0]
                                    if fit_function(args, source_path, tmp_path, target_path, path2, cpair, file_to_label, tokenizer, model, config, semantic_indices):
                                        adopted_seed.append(sub_file[0])
                                        if cpair not in semantic_seeds:
                                            semantic_seeds[cpair] = [[sub_file[0], sub_file[1], item[0]]]
                                        else:
                                            semantic_seeds[cpair].append([sub_file[0], sub_file[1], item[0]])
                            else:
                                if sub_file[0][:-(len(m_seed[-1])+1)] in adopted_seed:
                                    source_path = './mutated_candidates/mutated_code'+sub_file[0][:-(len(m_seed[-2])+1+len(m_seed[-1])+1)]+'/'+sub_file[1]
                                    tmp_path = './mutated_candidates/mutated_code'+sub_file[0][:-(len(m_seed[-1])+1)]+'/'+sub_file[1]
                                    target_path = './mutated_candidates/mutated_code'+sub_file[0]+'/'+sub_file[1]
                                    path2 = './code2/'+item[0]
                                    if fit_function(args, source_path, tmp_path, target_path, path2, cpair, file_to_label, tokenizer, model, config, semantic_indices):
                                        adopted_seed.append(sub_file[0])
                                        if cpair not in semantic_seeds:
                                            semantic_seeds[cpair] = [[sub_file[0], sub_file[1], item[0]]]
                                        else:
                                            semantic_seeds[cpair].append([sub_file[0], sub_file[1], item[0]])

            # surface_seeds = {}
            # syntax_seeds = {}
            # semantic_seeds = {}
            # for file in tqdm.tqdm(surface_files):
            #     t_file = file.split('-')[0] + '.c'
            #     for k, v in surface_mutated_files.items():
            #         get_initial_seeds(file, t_file, k, v, surface_seeds)
            #     for k, v in syntax_mutated_files.items():
            #         get_initial_seeds(file, t_file, k, v, syntax_seeds)
            #     for k, v in semantic_mutated_files.items():
            #         get_initial_seeds(file, t_file, k, v, semantic_seeds)
            with open('./surface_seeds.json', 'w') as fp:
                json.dump(surface_seeds, fp)
            with open('./syntax_seeds.json', 'w') as fp:
                json.dump(syntax_seeds, fp)
            with open('./semantic_seeds.json', 'w') as fp:
                json.dump(semantic_seeds, fp)

        # print(initial_seeds)
        # for k, v in initial_seeds.items():
        #     if k == 'a553c6a347d3d28d7ee44c3df3d5c4ee780dba23_10398.c' or k.split('-')[0]+'.c' == 'a553c6a347d3d28d7ee44c3df3d5c4ee780dba23_10398.c':
        #         print(k)
    surface_candidates = {}
    syntax_candidates = {}
    semantic_candidates = {}
    # atten_candidates = {}
    get_candidates(args, surface_seeds, tokenizer, model, config, surface_candidates, surface_indices, file_to_label)   
    get_candidates(args, syntax_seeds, tokenizer, model, config, syntax_candidates, syntax_indices, file_to_label)
    get_candidates(args, semantic_seeds, tokenizer, model, config, semantic_candidates, semantic_indices, file_to_label)

    # with open('surface_candidates.json', 'w') as fp:
    #     json.dump(surface_candidates, fp)
    print(len(surface_candidates), len(syntax_candidates), len(semantic_candidates)) 
    return surface_candidates, syntax_candidates, semantic_candidates, sub_query
    

def testing(surface_candidates, syntax_candidates, semantic_candidates, file_to_label, sub_query):
    query = 0.0
    testcases_num = 0.0
    success = 0.0
    

    '''surface层面的扰动'''
    syntax_initial_files = []
    for fname, features in tqdm.tqdm(surface_candidates.items()):
         for i, sub_features in enumerate(features):
            predict = sub_features[-1]
            query += 1.0
            if predict != int(file_to_label[fname]):
                testcases_num += 1.0
                success += 1.0
                break
            elif fname not in syntax_initial_files:
                syntax_initial_files.append(fname)

    '''syntax层面的扰动'''
    semantic_initial_files = []
    for fname in tqdm.tqdm(syntax_initial_files):
        if fname in syntax_candidates:
            for i, sub_features in enumerate(syntax_candidates[fname]):
                predict = sub_features[-1]
                query += 1.0
                if predict != int(file_to_label[fname]):
                    testcases_num += 1.0
                    success += 1.0
                    break
                elif fname not in semantic_initial_files:
                    semantic_initial_files.append(fname)
        else:
            semantic_initial_files.append(fname)
    
    '''semantic层面的扰动'''
    for fname in tqdm.tqdm(semantic_initial_files):
        if fname in semantic_candidates:
            for i, sub_features in enumerate(semantic_candidates[fname]):
                predict = sub_features[-1]
                query += 1.0
                if predict != int(file_to_label[fname]):
                    testcases_num += 1.0
                    success += 1.0
                    break
                elif i == 0:
                    testcases_num += 1.0
    
    attacking_file.close()
    print('[+] Results: query numbr:{}, success examples:{}, all testcases:{}, attacking successful rate:{}'.format(query+sub_query, success, testcases_num, success / testcases_num))
    # testing_result.close()
            


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default='./dataset/train_sampled.txt', type=str)
    parser.add_argument("--output_dir", default='./saved_models', type=str)
    ## Other parameters
    parser.add_argument("--eval_data_file", default='./dataset/valid_sampled.txt', type=str)
    parser.add_argument("--test_data_file", default='./dataset/test_sampled.txt', type=str)            
    parser.add_argument("--model_type", default="roberta", type=str)
    parser.add_argument("--model_name_or_path", default='microsoft/graphcodebert-base', type=str)
    parser.add_argument("--mlm", action='store_true')
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--config_name", default="microsoft/graphcodebert-base", type=str)
    parser.add_argument("--tokenizer_name", default="microsoft/graphcodebert-base", type=str)
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--code_length", default=512, type=int) 
    parser.add_argument("--data_flow_length", default=128, type=int) 
    parser.add_argument("--do_test", action='store_false')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--learning_rate", default=5e-5)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=1.0, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu
    # Setup logging

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=1
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)

    model=Model(model,config,tokenizer,args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab
    
    checkpoint_prefix = 'checkpoint-best-acc/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
    model.load_state_dict(torch.load(output_dir))                  
    model.to(args.device)
    # for name, params in model.named_parameters():
    #     print(name, params)

    file_to_label = {}
    file_to_pairs = {}
    with open('./original_cases.txt','r') as fp:
        all_lines = fp.read().strip().split('\n')
        for line in tqdm.tqdm(all_lines):
            id1 = line.split('  ')[0]
            id2 = line.split('  ')[1]
            label = line.split('  ')[-1]
            file_to_label[id1+'__'+id2] = label
            if id1 not in file_to_pairs:
                file_to_pairs[id1] = [[id2, label]]
            else:
                file_to_pairs[id1].append([id2, label])
    
    mutated_files_list = {}
    for index in tqdm.tqdm(mutation_index.values()):
        source_dir = './mutated_candidates/true_mutated'+index+'.txt'
        mutated_list = get_file_list(source_dir, [])
        for file in mutated_list:
            if len(file.split('-')) == 1:
                s_name = file
            else:
                s_name = file.split('-')[0]+'.c'
            if s_name not in mutated_files_list:
                mutated_files_list[s_name] = [[index, file]]
            else:
                mutated_files_list[s_name].append([index, file])
        # mutated_files_list[index] = mutated_list
    with open('mutated_files_list.json', 'w') as fp:
        json.dump(mutated_files_list, fp)
    # fuzzer(config, args, model, tokenizer, file_to_label)
    surface_candidates, syntax_candidates, semantic_candidates, sub_query = fuzzer(config, args, model, tokenizer, file_to_pairs, file_to_label, mutated_files_list)
    testing(surface_candidates, syntax_candidates, semantic_candidates, file_to_label, sub_query)

    # # atten_candidates, sub_query = fuzzer('e_distance', config, args, model, file_to_label)
    # # testing(2, atten_candidates, file_to_label, sub_query)

if __name__ == "__main__":
    main()