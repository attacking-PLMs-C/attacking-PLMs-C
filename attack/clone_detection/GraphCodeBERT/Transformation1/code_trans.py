import os
import sys
import json
import tqdm
import re
from collections import Counter
import operator
import random
import argparse
import torch.nn.functional as F
import pandas as pd
import subprocess
import itertools
from importlib import import_module
from transformers import RobertaTokenizer
from multiprocessing import Pool
sys.path.append('..')
from utils import *

# def get_file_path(root_path, file_list):
#     PATH = os.listdir(root_path)
#     for path in PATH:
#         # print(path)
#         co_path = os.path.join(root_path, path)
#         if os.path.isfile(co_path):
#             file_list.append(co_path)
#         elif os.path.isdir(co_path):
#             get_file_path(co_path, file_list)
#     return file_list

# def get_file_list(root_path, file_list):
#     with open(root_path, 'r') as fp:
#         all_lines = fp.read().strip().split('\n')
#         for line in all_lines:
#             file_list.append(line)
#     return file_list

def get_data(code_data):
    code_data = re.sub(r"(\s)+", '', code_data).strip()  # [\n ]+
    return code_data

def mutation_count(source_path1, source_path2, target_path):
    all_files = []
    mutated_files = []
    get_file_path(source_path1, all_files)
    get_file_path(source_path2, mutated_files)
    print(len(mutated_files))
    print(len(all_files))

    bar = tqdm.tqdm(mutated_files)
    with open(target_path, 'w') as fp:
        for path in bar:
            fname = path.split('/')[-1]
            if source_path1 == '../initial_code':
                s_path = source_path1 + '/' + fname.split('-')[0] + '.c'
            else:
                s_path = source_path1 + '/' + fname
            f_mutated = open(path,'r').read().strip()
            f_mutated = get_data(f_mutated)
            f_source = open(s_path, 'r').read().strip()
            f_source = get_data(f_source)
            if f_mutated != f_source:
                fp.write(path.split('/')[-1] + '\n')
            bar.update()
        bar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default='../dataset/train_sampled.txt', type=str)
    parser.add_argument("--output_dir", default='../saved_models', type=str)
    ## Other parameters
    parser.add_argument("--eval_data_file", default='../dataset/valid_sampled.txt', type=str)
    parser.add_argument("--test_data_file", default='../dataset/test_sampled.txt', type=str)            
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

    # file_path_list = []
    # get_file_path('../initial_code', file_path_list)
    # bar = tqdm.tqdm(total=len(file_path_list))
    # for i, path in  enumerate(file_path_list):
    #     fname = path.split('/')[-1]
    #     fid = fname.split('.')[0]
    #     subprocess.check_output('./runner1.sh ' + path + ' ' + fname + ' 1 ' + fid, shell=True)
    #     bar.update()
    # bar.close()
    # mutation_count('../initial_code', '../mutated_candidates/mutated_code1', '../mutated_candidates/true_mutated.txt')

    file_to_label = {}
    file_to_pairs = {}
    with open('../original_cases.txt','r') as fp:
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
    
    model.eval()
    tmp = []
    surface_indices = torch.tensor([5]).to(args.device)
    syntax_indices = torch.tensor([5,6,7]).to(args.device)
    semantic_indices = torch.tensor([8,9,10]).to(args.device)
    get_file_list('../mutated_candidates/true_mutated.txt', tmp)

    code_rename = []
    sub_query = 0.0
    variable_candidates = {}
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

        file_to_candidates = {}
        for file in tqdm.tqdm(code_rename):
            s_name = file.split('-')[0] + '.c'
            s_path = '../initial_code/' + s_name
            s_code = open(s_path, 'r').read().strip()
            for item in file_to_pairs[s_name]:
                s_path2 = '../code2/' + item[0]
                s_code2 = open(s_path2, 'r').read().strip()
                s_data = TextDataset(tokenizer, args, s_code, s_code2, s_name+'__'+item[0], int(file_to_label[s_name+'__'+item[0]]))
                s_dataloader = DataLoader(s_data, batch_size=1)
                for batch in s_dataloader:
                    input_ids = batch[0].to(args.device)
                    atten_mask = batch[1].to(args.device)
                    position_idx = batch[2].to(args.device)
                    label = batch[3].to(args.device)
                    idx = batch[4]
                    _, logit, s_hiddenout, s_attentionout = model(input_ids, atten_mask, position_idx, label)
                    pred = int(logit.cpu().numpy()>0.5)
                angle_out = torch.index_select(s_hiddenout, dim=1, index=surface_indices)[:,:,0,:]
                E_out = torch.index_select(s_attentionout, dim=1, index=surface_indices-1)
                max_distance1 = 0.0
                max_distance2 = 0.0
                best_cand = ''
                tag1 = True
                for sub_cand in variable_candidates[file]:
                    sub_path = '../mutated_candidates/mutated_code1/' + sub_cand
                    sub_code = open(sub_path, 'r').read().strip()
                    sub_data = TextDataset(tokenizer, args, sub_code, s_code2, s_name+'__'+item[0], int(file_to_label[s_name+'__'+item[0]]))
                    sub_dataloader = DataLoader(sub_data, batch_size=1)
                    for batch in sub_dataloader:
                        input_ids = batch[0].to(args.device)
                        atten_mask = batch[1].to(args.device)
                        position_idx = batch[2].to(args.device)
                        label = batch[3].to(args.device)
                        idx = batch[4]
                        _, sub_logit, sub_hiddenout, sub_attentionout = model(input_ids, atten_mask, position_idx, label)
                        lab = int(sub_logit.cpu().numpy()>0.5)
                    sub_angle_out = torch.index_select(sub_hiddenout, dim=1, index=surface_indices)[:,:,0,:]
                    sub_E_out = torch.index_select(sub_attentionout, dim=1, index=surface_indices-1)
                    if lab != pred:
                        best_cand = sub_cand
                        break
                    metric1 = F.cosine_similarity(angle_out, sub_angle_out, dim=-1)
                    metric1 = torch.acos(metric1) * 180 / 3.1415926
                    metric1 = torch.mean(metric1)
                    metric2 = torch.norm(E_out-sub_E_out, p=2)
                    if metric1 <= 0.0 or metric2 <= 0.0:
                        tag1 = False
                        break
                    if max_distance1 < metric1 or max_distance2 < metric2:
                        max_distance1 = metric1
                        max_distance2 = metric2
                        best_cand = sub_cand
                if tag1 == False:
                    continue
                '''形成每个克隆对和其candidates间的字典'''
                if s_name not in file_to_candidates:
                    file_to_candidates[s_name+'__'+item[0]] = [best_cand]
                elif s_name in file_to_candidates:
                    file_to_candidates[s_name+'__'+item[0]].append(best_cand)

        true_mutated1 = open('../mutated_candidates/true_mutated1.txt', 'w')
        surface_files = []
        for fname, cand_list in tqdm.tqdm(file_to_candidates.items()):
            s_name1, s_name2 = fname.split('__')[0], fname.split('__')[-1]
            s_name1 = s_name1.split('-')[0] + '.c'
            for i in range(len(cand_list)):
                if cand_list[i] == '':
                    continue
                sub_query += 1.0
                if i == len(cand_list) - 1:
                    true_mutated1.write(cand_list[i] + '\n')
                    surface_files.append([cand_list[i],s_name2])
                    break
                f_path = '../mutated_candidates/mutated_code1/' + cand_list[i]
                f_code = open(f_path, 'r').read().strip()
                f_path2 = '../code2/' + s_name2
                f_code2 = open(f_path2, 'r').read().strip()
                f_data = TextDataset(tokenizer, args, f_code, f_code2, fname, int(file_to_label[fname]))
                f_dataloader = DataLoader(f_data, batch_size=1)
                for batch in f_dataloader:
                    input_ids = batch[0].to(args.device)
                    atten_mask = batch[1].to(args.device)
                    position_idx = batch[2].to(args.device)
                    label = batch[3].to(args.device)
                    idx = batch[4]
                    _, f_logit, f_hiddenout, f_attentionout = model(input_ids, atten_mask, position_idx, label)
                    lab = int(f_logit.cpu().numpy()>0.5)
                if lab != int(file_to_label[fname]):
                    true_mutated1.write(cand_list[i] + '\n')
                    surface_files.append([cand_list[i],s_name2])
                    break
        true_mutated1.close()
        with open('../surface_files.json', 'w') as fp:
            json.dump(surface_files, fp)
        print('sub_query:{}'.format(sub_query))