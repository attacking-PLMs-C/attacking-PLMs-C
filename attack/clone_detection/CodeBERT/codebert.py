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
from torch.nn import DataParallel
from dataset_iter import DatasetIterdtor
sys.path.append('.')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    total_losses = 0
    total_labels = np.array([], dtype=int)
    total_predicts = np.array([], dtype=int)
    label_list = list(str(i) for i in range(2))
    with torch.no_grad():
        for id_, s, label in tqdm.tqdm(val_iter):
            s = s.cuda()
            label = label.cuda()
            outs = model(s)
            loss = F.cross_entropy(outs, label)
            total_losses += loss
            label = label.data.cpu().numpy()
            predict = torch.max(outs.data, 1)[1].cpu().numpy()
            total_labels = np.append(total_labels, label)
            total_predicts = np.append(total_predicts, predict)

    acc = metrics.accuracy_score(total_labels, total_predicts)
    if test:
        precision = metrics.precision_score(total_labels, total_predicts)
        recall = metrics.recall_score(total_labels, total_predicts)
        f1 = metrics.f1_score(total_labels, total_predicts)
        report = metrics.classification_report(total_labels, total_predicts, target_names=label_list, digits=4)
        confusion = metrics.confusion_matrix(total_labels, total_predicts)
        return acc, total_losses / len(val_iter), precision, recall, f1, report, confusion
    return acc, total_losses / len(val_iter)


def train(config, model, train_iter, val_iter, test_iter):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_loss = float(10)
    last_epoch = 0
    dev_best_acc = float(0)
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step()
        n_batch = 0
        train_losses = 0
        for id_, s, label in tqdm.tqdm(train_iter):
            model.zero_grad()
            s = s.cuda()
            label = label.cuda()
            outs = model(s)
            # model.zero_grad()
            batch_losses = F.cross_entropy(outs, label)
            train_losses += batch_losses.item()
            batch_losses.backward()
            optimizer.step()
            n_batch += 1
        if n_batch % len(train_iter) == 0:
        # if epoch % 1 == 0:  # best at 360
            true = label.data.cpu()
            predict = torch.max(outs.data, 1)[1].cpu()
            train_acc = metrics.accuracy_score(true, predict)
            dev_acc, dev_loss = evaluate(config, model, val_iter)
            train_loss = train_losses / len(train_iter)
            if dev_acc > dev_best_acc:
                dev_best_loss = dev_loss
                torch.save(model.module.state_dict(), config.save_path)
                
            msg = 'Epoch: {0:>1},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}'
            # msg > 表示在两者之间增加空格
            print(msg.format((epoch+1), train_loss, train_acc, dev_loss, dev_acc))
            model.train()
    test(config, model.module, test_iter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='./data/train_set_token.pkl', help='input train dataset')
    parser.add_argument('--val_data', type=str, default='./data/val_set_token.pkl', help='input validation dataset')
    parser.add_argument('--test_data', type=str, default='./data/test_set_token.pkl', help='input test dataset')
    parser.add_argument('--model', type=str, default='CodeBERT', help='model name')
    parser.add_argument('--train_eval', type=str, default='train')
    args = parser.parse_args()

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

    # args.train_data = './data/train_set_token.pkl'
    # args.test_data = './data/test_set_token.pkl'
    train_dataset = pkl.load(open(args.train_data, 'rb'))
    val_dataset = pkl.load(open(args.val_data, 'rb'))
    # print(val_dataset)
    test_dataset = pkl.load(open(args.test_data, 'rb'))
    train_iter = DatasetIterdtor(train_dataset, config.batch_size, device)
    val_iter = DatasetIterdtor(val_dataset, config.batch_size, device)
    test_iter = DatasetIterdtor(test_dataset, config.batch_size, device)
    model = X.CodeBERT(config)
    model = model.to(device)
    # device_ids = [0, 1]
    if args.train_eval == 'train':
        model = DataParallel(model)
        train(config, model, train_iter, val_iter, test_iter)
    elif args.train_eval == 'test':
        model.load_state_dict(torch.load(config.save_path))
        # model.load_state_dict(torch.load('./save_dict/CodeBERT.ckpt'))
        acc, test_loss, p, r, f1, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
        msg = 'Test acc: {0:>6.2%}, Test precision: {1:>6.2%}, Test recall: {2:>6.2%}, Test f1: {3:>6.2%}'
        print(msg.format(acc, p, r, f1))
        # # print("Precision, Recall and F1-Score...")
        # # print(test_report)
        # # print("Confusion Matrix...")
        # # print(test_confusion)