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
from utils import *
import pandas as pd
import warnings
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from importlib import import_module
import operator
from torch.nn import DataParallel
sys.path.append('.')

surface_mutants = {'0': '1'}
syntax_mutants = {'1': '1_12', '2': '12'}
# syntax_mutants = {'1': '2', '2': '7', '4': '1_2', '5': '1_7', '7': '1_2_7', '11': '2_7'}
semantic_mutants = {'3': '2', '4': '3', '5': '7', '6': '1_2', '7': '1_3', '8': '1_7', '9': '1_2_3', '10': '1_2_7', '11': '1_3_7', '12': '1_2_3_7', '13': '12_2', '14': '12_3', '15': '12_7',
                      '16': '12_2_3', '17': '12_2_7', '18': '12_3_7', '19': '12_2_3_7', '20': '1_12_2', '21': '1_12_3', '22': '1_12_7', '23': '1_12_2_3', '24': '1_12_2_7', '25': '1_12_3_7',
                      '26': '1_12_2_3_7', '27': '2_3', '28': '2_7', '29': '3_7', '30': '2_3_7'}
mutation_index = {'0': '1', '1': '1_12', '2': '12', '3': '2', '4': '3', '5': '7', '6': '1_2', '7': '1_3', '8': '1_7', '9': '1_2_3', '10': '1_2_7', '11': '1_3_7', '12': '1_2_3_7', '13': '12_2', '14': '12_3', '15': '12_7',
                      '16': '12_2_3', '17': '12_2_7', '18': '12_3_7', '19': '12_2_3_7', '20': '1_12_2', '21': '1_12_3', '22': '1_12_7', '23': '1_12_2_3', '24': '1_12_2_7', '25': '1_12_3_7',
                      '26': '1_12_2_3_7', '27': '2_3', '28': '2_7', '29': '3_7', '30': '2_3_7'}
# semantic_mutants = {'3': '12', '6': '1_12', '8': '1_2_12', '9': '1_7_12', '10': '1_2_7_12', '12': '2_12', '13': '2_7_12', '14': '7_12'}


def get_candidates(initial_seeds, tokenizer, model, config, initial_candidates, indices, mutants):
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
                _, s_hiddenout, s_attentionout = get_prediction(s_code, s_code2, tokenizer, model, config)
                angle_out = torch.index_select(s_hiddenout, dim=1, index=indices)[:,:,0,:]
                E_out = torch.index_select(s_attentionout, dim=1, index=indices-1)
                f_path = './mutated_candidates/mutated_code'+sub_seed[0]+'/'+sub_seed[1]
                f_code = open(f_path, 'r').read().strip()
                f_predict, f_hiddenout, f_attentionout = get_prediction(f_code, s_code2, tokenizer, model, config)
                angle_out1 = torch.index_select(f_hiddenout, dim=1, index=indices)[:,:,0,:]
                E_out1 = torch.index_select(f_attentionout, dim=1, index=indices-1)
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

# def get_candidates(initial_seeds, tokenizer, model, config, initial_candidates, indices, mutants):
#     model.eval()
#     with torch.no_grad():
#         for fc, seed in tqdm.tqdm(initial_seeds.items()):
#             sub_features = {}
#             if len(fc.split('-')) == 3:
#                 s_path = './initial_code/' + fc.split('-')[0] + '.c'
#                 s_code = open(s_path, 'r').read().strip() 
#                 for sub_seed in seed:
#                     s_path2 = './code2/' + sub_seed[1]
#                     s_code2 = open(s_path2, 'r').read().strip()
#                     _, s_hiddenout, s_attentionout = get_prediction(s_code, s_code2, tokenizer, model, config)
#                     angle_out = torch.index_select(s_hiddenout, dim=1, index=indices)[:,:,0,:]
#                     E_out = torch.index_select(s_attentionout, dim=1, index=indices-1)
#                     f_path = './mutated_candidates/mutated_code' + mutants[sub_seed[0]] + '/' + fc
#                     f_code = open(f_path, 'r').read().strip()
#                     f_predict, f_hiddenout, f_attentionout = get_prediction(f_code, s_code2, tokenizer, model, config)
#                     angle_out1 = torch.index_select(f_hiddenout, dim=1, index=indices)[:,:,0,:]
#                     E_out1 = torch.index_select(f_attentionout, dim=1, index=indices-1)
#                     metric = F.cosine_similarity(angle_out, angle_out1, dim=-1)
#                     metric = torch.acos(metric) * 180 / 3.1415926
#                     metric = torch.mean(metric)
#                     if metric <= 0.0:
#                         continue
#                     feat1 = [f_path, metric.item(), f_predict]
#                     pair_id = fc.split('-')[0] + '.c' + '__' + sub_seed[1]
#                     if pair_id not in sub_features:
#                         sub_features[pair_id] = [feat1]
#                     else:
#                         sub_features[pair_id].append(feat1)
                        
#                 if fc.split('-')[0]+'.c' in initial_seeds:
#                     for sub_seed in initial_seeds[fc.split('-')[0]+'.c']:
#                         f_path1 = './mutated_candidates/mutated_code' + mutants[sub_seed[0]] + '/' + fc.split('-')[0] + '.c'
#                         f_code1 = open(f_path1, 'r').read().strip()
#                         f_path2 = './code2/' + sub_seed[1]
#                         f_code2 = open(f_path2, 'r').read().strip()
#                         f_predict1, f_hiddenout1, f_attentionout1 = get_prediction(f_code1, f_code2, tokenizer, model, config)
#                         angle_out2 = torch.index_select(f_hiddenout1, dim=1, index=indices)[:,:,0,:]
#                         E_out2 = torch.index_select(f_attentionout1, dim=1, index=indices-1)
#                         metric1 = F.cosine_similarity(angle_out, angle_out2, dim=-1)
#                         metric1 = torch.acos(metric1) * 180 / 3.1415926
#                         metric1 = torch.mean(metric1)
#                         if metric1 <= 0.0:
#                             continue
#                         pair_id = fc.split('-')[0] + '.c' + '__' + sub_seed[1]
#                         feat2 = [f_path1, metric1.item(), f_predict1]
#                         if pair_id not in sub_features:
#                             sub_features[pair_id] = [feat2]
#                         else:
#                             sub_features[pair_id].append(feat2)
                
#                 for pi, sf in sub_features.items():
#                     initial_candidates[pi] = sorted(sf, key=operator.itemgetter(1), reverse=True)
#     return initial_candidates

def get_initial_seeds(file, t_file, k, v, initial_seeds):
    if file[0] in v and file[0] not in initial_seeds:
        initial_seeds[file[0]] = [[k,file[1]]]
    elif t_file in v and t_file not in initial_seeds:
        initial_seeds[t_file] = [[k,file[1]]]
    elif file[0] in v and file[0] in initial_seeds:
        initial_seeds[file[0]].append([k,file[1]])
    elif t_file in v and t_file in initial_seeds:
        initial_seeds[t_file].append([k,file[1]])
    return initial_seeds

# def fit_function(code_name2, m_file, source_path, tmp_path, tokenizer, model, config, liguistic_mutants, liguistic_indices, idx, tag=False):
#     if tag == True:
#         fname = m_file
#     else:
#         fname = m_file.split('-')[0] + '.c'
#     s_code = open(source_path+fname, 'r').read().strip()
#     code2 = open('./code2/'+code_name2, 'r').read().strip()
#     _, s_hiddenout, s_attentionout = get_prediction(s_code, code2, tokenizer, model, config)
#     angle_out = torch.index_select(s_hiddenout, dim=1, index=liguistic_indices)[:,:,0,:]
#     E_out = torch.index_select(s_attentionout, dim=1, index=liguistic_indices-1)

#     tmp_code = open(tmp_path+'/'+m_file, 'r').read().strip()
#     _, tmp_hiddenout, tmp_attentionout = get_prediction(tmp_code, code2, tokenizer, model, config)
#     tmp_angle_out = torch.index_select(tmp_hiddenout, dim=1, index=liguistic_indices)[:,:,0,:]
#     tmp_E_out = torch.index_select(tmp_attentionout, dim=1, index=liguistic_indices-1)

#     t_code = open('./mutated_candidates/mutated_code'+liguistic_mutants[str(idx)]+'/'+m_file, 'r').read().strip()
#     _, t_hiddenout, t_attentionout = get_prediction(t_code, code2, tokenizer, model, config)
#     t_angle_out = torch.index_select(t_hiddenout, dim=1, index=liguistic_indices)[:,:,0,:]
#     t_E_out = torch.index_select(t_attentionout, dim=1, index=liguistic_indices-1)

#     cs_value = F.cosine_similarity(angle_out, tmp_angle_out, dim=-1)
#     adho = torch.acos(cs_value) * 180 / 3.1415926
#     adho = torch.mean(adho)
#     edao = torch.norm(E_out-tmp_E_out, p=2)

#     cs_value1 = F.cosine_similarity(tmp_angle_out, t_angle_out, dim=-1)
#     adho1 = torch.acos(cs_value1) * 180 / 3.1415926
#     adho1 = torch.mean(adho1)
#     edao1 = torch.norm(tmp_E_out-t_E_out, p=2)

#     if adho1 >= adho and edao1 >= edao:
#         return True
#     else:
#         return False

def fit_function(source_path, tmp_path, target_path, path2, tokenizer, model, config, linguistic_indices):
    code2 = open(path2, 'r').read().strip()
    s_code = open(source_path, 'r').read().strip()
    _, s_hiddenout, s_attentionout = get_prediction(s_code, code2, tokenizer, model, config)
    angle_out = torch.index_select(s_hiddenout, dim=1, index=linguistic_indices)[:,:,0,:]
    E_out = torch.index_select(s_attentionout, dim=1, index=linguistic_indices-1)

    tmp_code = open(tmp_path, 'r').read().strip()
    _, tmp_hiddenout, tmp_attentionout = get_prediction(tmp_code, code2, tokenizer, model, config)
    tmp_angle_out = torch.index_select(tmp_hiddenout, dim=1, index=linguistic_indices)[:,:,0,:]
    tmp_E_out = torch.index_select(tmp_attentionout, dim=1, index=linguistic_indices-1)

    t_code = open(target_path, 'r').read().strip()
    _, t_hiddenout, t_attentionout = get_prediction(t_code, code2, tokenizer, model, config)
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

def fuzzer(config, args, model, file_to_pairs, file_to_label, mutated_files_list):
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
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    
    surface_indices = torch.tensor([5]).to(config.device)
    syntax_indices = torch.tensor([5,6,7]).to(config.device)
    semantic_indices = torch.tensor([8,9,10]).to(config.device)
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
            #     for item in file_to_pairs[s_name]:
            #         s_path2 = './code2/' + item[0]
            #         s_code2 = open(s_path2, 'r').read().strip()
            #         lab, s_hiddenout, s_attentionout = get_prediction(s_code, s_code2, tokenizer, model, config)
            #         angle_out = torch.index_select(s_hiddenout, dim=1, index=surface_indices)[:,:,0,:]
            #         E_out = torch.index_select(s_attentionout, dim=1, index=surface_indices-1)
            #         max_distance = 0.0
            #         best_cand = ''
            #         tag1 = True
            #         for sub_cand in variable_candidates[file]:
            #             sub_path = './mutated_candidates/mutated_code1/' + sub_cand
            #             sub_code = open(sub_path, 'r').read().strip()
            #             pred, sub_hiddenout, sub_attentionout = get_prediction(sub_code, s_code2, tokenizer, model, config)
            #             sub_angle_out = torch.index_select(sub_hiddenout, dim=1, index=surface_indices)[:,:,0,:]
            #             sub_E_out = torch.index_select(sub_attentionout, dim=1, index=surface_indices-1)
            #             if lab != pred:
            #                 best_cand = sub_cand
            #                 break
            #             if args.d_type == 'a_distance':
            #                 metric = F.cosine_similarity(angle_out, sub_angle_out, dim=-1)
            #                 metric = torch.acos(metric) * 180 / 3.1415926
            #                 metric = torch.mean(metric)
            #             else:
            #                 metric = torch.norm(E_out-sub_E_out, p=2)
            #             if metric <= 0.0:
            #                 tag1 = False
            #                 break
            #             if max_distance < metric:
            #                 max_distance = metric
            #                 best_cand = sub_cand
            #         if tag1 == False:
            #             continue
            #         '''形成每个克隆对和其candidates间的字典'''
            #         if s_name not in file_to_candidates:
            #             file_to_candidates[s_name+'__'+item[0]] = [best_cand]
            #         elif s_name in file_to_candidates:
            #             file_to_candidates[s_name+'__'+item[0]].append(best_cand)
            #     # with open('./file_to_candidates.json', 'w') as fp:
            #     #     json.dump(file_to_candidates, fp)
            #     # print(file_to_candidates)

            # for fname, cand_list in tqdm.tqdm(file_to_candidates.items()):
            #     s_name1, s_name2 = fname.split('__')[0], fname.split('__')[-1]
            #     s_name1 = s_name1.split('-')[0] + '.c'
            #     for i in range(len(cand_list)):
            #         if cand_list[i] == '':
            #             continue
            #         sub_query += 1.0
            #         if i == len(cand_list) - 1:
            #             surface_files.append([cand_list[i],s_name2])
            #             break
            #         f_path = './mutated_candidates/mutated_code1/' + cand_list[i][0]
            #         f_code = open(f_path, 'r').read().strip()
            #         f_path2 = './code2/' + s_name2
            #         f_code2 = open(f_path2, 'r').read().strip()
            #         predict, _, _ = get_prediction(f_code, f_code2, tokenizer, model, config)
            #         if predict != int(file_to_label[fname]):
            #             surface_files.append([cand_list[i],s_name2])
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
                                    if fit_function(source_path, tmp_path, target_path, path2, tokenizer, model, config, semantic_indices):
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
                                    if fit_function(source_path, tmp_path, target_path, path2, tokenizer, model, config, semantic_indices):
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
                                    if fit_function(source_path, tmp_path, target_path, path2, tokenizer, model, config, semantic_indices):
                                        adopted_seed.append(sub_file[0])
                                        if cpair not in semantic_seeds:
                                            semantic_seeds[cpair] = [[sub_file[0], sub_file[1], item[0]]]
                                        else:
                                            semantic_seeds[cpair].append([sub_file[0], sub_file[1], item[0]])


            # fit_function(code_name2, m_file, source_path, tmp_path, tokenizer, model, config, semantic_mutants, semantic_indices, i, True)

            # mutatant_seed = {'1': '0', '1_12': '1', '12': '2'}
            # for key, value in semantic_mutants.items():
            #     mutatant_seed[value] = key
            
            # all_mutated_files = {}
            # for i in tqdm.tqdm(range(31)):
            #     if str(i) in surface_mutants:
            #         source_path = './mutated_candidates/true_mutated' + surface_mutants[str(i)] + '.txt'
            #         surface_mutated_files[str(i)] = get_file_list(source_path, [])
            #         all_mutated_files[str(i)] = get_file_list(source_path, [])
            #     elif str(i) in syntax_mutants:
            #         m_seeds = syntax_mutants[str(i)].split('_')
            #         if len(m_seeds) == 1:
            #             source_path = './mutated_candidates/true_mutated' + syntax_mutants[str(i)] + '.txt'
            #             syntax_mutated_files[str(i)] = get_file_list(source_path, [])
            #             all_mutated_files[str(i)] = get_file_list(source_path, [])
            #         else:
            #             syntax_mutated_files[str(i)] = []
            #             all_mutated_files[str(i)] = []
            #             target_path = './mutated_candidates/true_mutated' + syntax_mutants[str(i)] + '.txt'
            #             target_mutants = get_file_list(target_path, [])
            #             source_path = './initial_code/'
            #             tmp_path = './mutated_candidates/mutated_code' + syntax_mutants[str(i)][:-(len(m_seeds[-1])+1)]
            #             for m_file in target_mutants:
            #                 if m_file in all_mutated_files[mutatant_seed[syntax_mutants[str(i)][:-(len(m_seeds[-1])+1)]]]:
            #                     if len(m_file.split('-')) == 1:
            #                         s_name = m_file
            #                     else:
            #                         s_name = m_file.split('-')[0] + '.c'
            #                     code_name2 = file_to_pairs[s_name]
            #                     if fit_function(code_name2, m_file, source_path, tmp_path, tokenizer, model, config, syntax_mutants, syntax_indices, i):
            #                         syntax_mutated_files[str(i)].append(m_file)
            #                         all_mutated_files[str(i)].append(m_file)

            #     elif str(i) in semantic_mutants:
            #         m_seeds = semantic_mutants[str(i)].split('_')
            #         if len(m_seeds) == 1:
            #             source_path = './mutated_candidates/true_mutated' + semantic_mutants[str(i)] + '.txt'
            #             semantic_mutated_files[str(i)] = get_file_list(source_path, [])
            #             all_mutated_files[str(i)] = get_file_list(source_path, [])
            #         elif len(m_seeds) == 2:
            #             semantic_mutated_files[str(i)] = []
            #             all_mutated_files[str(i)] = []
            #             target_path = './mutated_candidates/true_mutated' + semantic_mutants[str(i)] + '.txt'
            #             target_mutants = get_file_list(target_path, [])
            #             source_path = './initial_code/'
            #             tmp_path = './mutated_candidates/mutated_code' + semantic_mutants[str(i)][:-(len(m_seeds[-1])+1)]
            #             for m_file in target_mutants:
            #                 if m_file in all_mutated_files[mutatant_seed[semantic_mutants[str(i)][:-(len(m_seeds[-1])+1)]]]:
            #                     if len(m_file.split('-')) == 1:
            #                         s_name = m_file
            #                     else:
            #                         s_name = m_file.split('-')[0] + '.c'
            #                     code_name2 = file_to_pairs[s_name]
            #                     if i <= 9 and fit_function(code_name2, m_file, source_path, tmp_path, tokenizer, model, config, semantic_mutants, semantic_indices, i):
            #                         semantic_mutated_files[str(i )].append(m_file)
            #                         all_mutated_files[str(i)].append(m_file)
            #                     elif i > 9 and fit_function(code_name2, m_file, source_path, tmp_path, tokenizer, model, config, semantic_mutants, semantic_indices, i, True):
            #                         semantic_mutated_files[str(i)].append(m_file)
            #                         all_mutated_files[str(i)].append(m_file)
            #         else:
            #             semantic_mutated_files[str(i)] = []
            #             all_mutated_files[str(i)] = []
            #             target_path = './mutated_candidates/true_mutated' + semantic_mutants[str(i)] + '.txt'
            #             target_mutants = get_file_list(target_path, [])
            #             source_path = './mutated_candidates/mutated_code' + semantic_mutants[str(i)][:-(len(m_seeds[-2])+1+len(m_seeds[-1])+1)] + '/'
            #             # source_path = './initial_code/'
            #             tmp_path = './mutated_candidates/mutated_code' + semantic_mutants[str(i)][:-(len(m_seeds[-1])+1)]
            #             for m_file in target_mutants:
            #                 if m_file in all_mutated_files[mutatant_seed[semantic_mutants[str(i)][:-(len(m_seeds[-1])+1)]]]:
            #                     if len(m_file.split('-')) == 1:
            #                         s_name = m_file
            #                     else:
            #                         s_name = m_file.split('-')[0] + '.c'
            #                     code_name2 = file_to_pairs[s_name]
            #                     if fit_function(code_name2, m_file, source_path, tmp_path, tokenizer, model, config, semantic_mutants, semantic_indices, i, True):
            #                         semantic_mutated_files[str(i)].append(m_file)
            #                         all_mutated_files[str(i)].append(m_file)
            # # # print(len(surface_files))
            # # for i in tqdm.tqdm(range(15)):
            # #     if str(i) in surface_mutants:
            # #         source_path = './mutated_candidates/true_mutated' + surface_mutants[str(i)] + '.txt'
            # #         surface_mutated_files[str(i)] = get_file_list(source_path, [])
            # #     elif str(i) in syntax_mutants:
            # #         source_path = './mutated_candidates/true_mutated' + syntax_mutants[str(i)] + '.txt'
            # #         syntax_mutated_files[str(i)] = get_file_list(source_path, [])
            # #     elif str(i) in semantic_mutants:
            # #         source_path = './mutated_candidates/true_mutated' + semantic_mutants[str(i)] + '.txt'
            # #         semantic_mutated_files[str(i)] = get_file_list(source_path, [])
            # # # print(true_mutated_files['14'])

            # surface_seeds = {}
            # syntax_seeds = {}
            # semantic_seeds = {}
            # for file in tqdm.tqdm(surface_files):
            #     t_file = file[0].split('-')[0] + '.c'
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
    print('[+] generating mutated candidates...')
    surface_candidates = {}
    syntax_candidates = {}
    semantic_candidates = {}
    # # atten_candidates = {}
    get_candidates(surface_seeds, tokenizer, model, config, surface_candidates, surface_indices, surface_mutants)   
    get_candidates(syntax_seeds, tokenizer, model, config, syntax_candidates, syntax_indices, syntax_mutants)
    get_candidates(semantic_seeds, tokenizer, model, config, semantic_candidates, semantic_indices, semantic_mutants)

    # with open('surface_candidates.json', 'w') as fp:
    #     json.dump(surface_candidates, fp)
    print(len(surface_candidates), len(syntax_candidates), len(semantic_candidates)) 
    return surface_candidates, syntax_candidates, semantic_candidates, sub_query
    

def testing(surface_candidates, syntax_candidates, semantic_candidates, file_to_label, sub_query):
    query = 0.0
    testcases_num = 0.0
    success = 0.0
    
    attacking_file = open('./advesarial_examples.txt', 'w')
    surface_examples = 0.0
    syntax_initial_files = []
    for fname, features in tqdm.tqdm(surface_candidates.items()):
         for i, sub_features in enumerate(features):
            predict = sub_features[-1]
            query += 1.0
            if predict != int(file_to_label[fname]):
                testcases_num += 1.0
                surface_examples += 1.0
                success += 1.0
                attacking_file.write(sub_features[0]+'\n')
                break
            elif fname not in syntax_initial_files:
                syntax_initial_files.append(fname)

    semantic_initial_files = []
    for fname in tqdm.tqdm(syntax_initial_files):
        if fname in syntax_candidates:
            for i, sub_features in enumerate(syntax_candidates[fname]):
                predict = sub_features[-1]
                query += 1.0
                if predict != int(file_to_label[fname]):
                    testcases_num += 1.0
                    success += 1.0
                    attacking_file.write(sub_features[0]+'\n')
                    break
                elif fname not in semantic_initial_files:
                    semantic_initial_files.append(fname)
        else:
            semantic_initial_files.append(fname)
    
    for fname in tqdm.tqdm(semantic_initial_files):
        if fname in semantic_candidates:
            for i, sub_features in enumerate(semantic_candidates[fname]):
                predict = sub_features[-1]
                query += 1.0
                if predict != int(file_to_label[fname]):
                    testcases_num += 1.0
                    success += 1.0
                    attacking_file.write(sub_features[0]+'\n')
                    break
                elif i == 0:
                    testcases_num += 1.0
    
    print('surface ASR:{}'.format(surface_examples / testcases_num))
    print('[+] Results: query numbr:{}, success examples:{}, all testcases:{}, attacking successful rate:{}'.format(query+sub_query, success, testcases_num, success / testcases_num))
    # testing_result.close()
            


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Encoder', help='model name')
    # parser.add_argument('--level', type=str, default='surface')
    parser.add_argument('--d_type', type=str, default='a_distance')
    args = parser.parse_args()

    model_name = args.model
    X = import_module('module.' + model_name)
    config = X.Config()
    device = config.device
    dir_name = './save_dict/'
    config.save_path = dir_name + 'CodeBERT.ckpt'

    model = X.Encoder(config)
    model = model.to(device)
    model.load_state_dict(torch.load(config.save_path))

    file_to_label = {}
    file_to_pairs = {}
    with open('original_cases.txt','r') as fp:
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

    # print(file_to_pairs)
    # pair_len = 0
    # cnt = 0
    # for k,v in file_to_pairs.items():
    #     cnt += len(v)
    #     if len(v) == 1:
    #         pair_len += 1
    # print(cnt, pair_len)

    # fuzzer(config, args, model, file_to_pairs, file_to_label, mutated_files_list)
    surface_candidates, syntax_candidates, semantic_candidates, sub_query = fuzzer(config, args, model, file_to_pairs, file_to_label, mutated_files_list)
    testing(surface_candidates, syntax_candidates, semantic_candidates, file_to_label, sub_query)

    # atten_candidates, sub_query = fuzzer('e_distance', config, args, model, file_to_label)
    # testing(2, atten_candidates, file_to_label, sub_query)

if __name__ == "__main__":
    main()