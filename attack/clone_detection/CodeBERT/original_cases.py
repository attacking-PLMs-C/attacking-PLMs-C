import pandas as pd
import warnings
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
import pickle as pkl
import tqdm
from importlib import import_module
import sys
import argparse
import subprocess
from torch.nn import DataParallel
from dataset_iter import DatasetIterdtor
sys.path.append('.')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    test_acc, test_loss, test_precision, test_recall, test_f1, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Acc: {0:>6.2%}, Test precision: {1:>6.2%}, Test recall: {2:>6.2%}, Test f1: {3:>6.2%}'
    print(msg.format(test_acc, test_precision, test_recall, test_f1))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)

def evaluate(config, model, val_iter, test=False):
    model.eval()
    total_id = []
    total_losses = 0
    total_labels = np.array([], dtype=int)
    total_predicts = np.array([], dtype=int)
    label_list = list(str(i) for i in range(2))
    with torch.no_grad():
        for id_, s, label in tqdm.tqdm(val_iter):
            fid = id_
            scode = s
            s = s.cuda()
            label = label.cuda()
            outs = model(s)
            loss = F.cross_entropy(outs, label)
            total_losses += loss
            label = label.detach().cpu().numpy()

            predict = torch.max(outs.detach(), 1)[1].cpu().numpy()
            total_labels = np.append(total_labels, label)
            total_id.extend(fid)
            total_predicts = np.append(total_predicts, predict)

    acc = metrics.accuracy_score(total_labels, total_predicts)
    with open('training_cases.txt', 'w') as fp:
        for i, _ in  enumerate(total_id):
            # print(len(_))
            id1_ = _.split('_')[0]
            id2_ = _.split('_')[-1]
            id1 = id1_.split('/')[1]+'_'+id1_.split('/')[-1]
            id2 = id2_.split('/')[1]+'_'+id2_.split('/')[-1]
            if total_labels[i] == total_predicts[i]:
                fp.write(id1 + '  ' + id2 + '  ' + str(total_labels[i]) + '  ' + str(total_predicts[i]) + '\n')
                # dn = _.split('.')[0].split('_')[-1]
                subprocess.check_output('cp ./dataset/code/' + id1 + '  ./initial_tcode/', shell=True)
                # subprocess.check_output('cp ./dataset/code/' + id2 + '  ./train_code2/', shell=True)

            
    precision = metrics.precision_score(total_labels, total_predicts)
    recall = metrics.recall_score(total_labels, total_predicts)
    f1 = metrics.f1_score(total_labels, total_predicts)
    report = metrics.classification_report(total_labels, total_predicts, target_names=label_list, digits=4)
    confusion = metrics.confusion_matrix(total_labels, total_predicts)
    return acc, total_losses / len(val_iter), precision, recall, f1, report, confusion
    # return acc, total_losses / len(val_iter)

def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='./data/train_set_token.pkl', help='input train dataset')
    parser.add_argument('--val_data', type=str, default='./data/val_set_token.pkl', help='input validation dataset')
    parser.add_argument('--test_data', type=str, default='./data/train_set_token.pkl', help='input test dataset')
    parser.add_argument('--model', type=str, default='CodeBERT', help='model name')
    args1 = parser.parse_args()
    return args1

args = get_parameters()

if __name__ == '__main__':
    model_name = args.model
    X = import_module('module.' + model_name)
    config = X.Config()
    # print('[+] loading vocabulary...')
    # code_vocab = json.load(open(args.p + '_code_vocab.json'))
    device = config.device
    # config.n_vocab = len(code_vocab)
    dir_name = './save_dict'
    config.save_path = dir_name + '/' + model_name + '.ckpt'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    train_dataset = pkl.load(open(args.train_data, 'rb'))
    val_dataset = pkl.load(open(args.val_data, 'rb'))
    test_dataset = pkl.load(open(args.test_data, 'rb'))
    train_iter = DatasetIterdtor(train_dataset, config.batch_size, device)
    val_iter = DatasetIterdtor(val_dataset, config.batch_size, device)
    test_iter = DatasetIterdtor(test_dataset, config.batch_size, device)
    model = X.CodeBERT(config)
    model = model.to(device)
    # device_ids = [0, 1]
    # model = DataParallel(model)
    # train(config, model, train_iter, val_iter, test_iter)
    model.load_state_dict(torch.load(config.save_path))
    # model.load_state_dict(torch.load('./save_dict/CodeBERT.ckpt'))
    acc, test_loss, p, r, f1, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test acc: {0:>6.2%}, Test precision: {1:>6.2%}, Test recall: {2:>6.2%}, Test f1: {3:>6.2%}'
    print(msg.format(acc, p, r, f1))
    # print("Precision, Recall and F1-Score...")
    # print(test_report)
    # print("Confusion Matrix...")
    # print(test_confusion)