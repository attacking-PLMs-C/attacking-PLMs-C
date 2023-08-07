from turtle import forward
import numpy as np
import math
import torch.nn.functional as F
from unicodedata import bidirectional
import torch
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn


class Config:
    '''模型参数配置'''
    def __init__(self):
        self.model_name = 'CodeBERT'
        self.layer = 12
        self.save_path = './save_dict/' + self.model_name + '.pt'  # 保存模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

        self.num_classes = 80  #类别数
        self.num_epochs = 5 #20
        # self.max_repoch = 3
        self.batch_size = 16  #mini_batch大小
        self.pad_size = 512  #每句话处理的长度大小（截取或填补）
        self.learning_rate = 2e-5  #学习率
        self.cb_embed = 768
        self.hidden_size = 256  #LSTM隐藏层
        self.num_layers = 2  #LSTM层数
    

class CodeBERT(nn.Module):
    def __init__(self, config):
        super(CodeBERT, self).__init__()
        self.codebert = RobertaModel.from_pretrained("microsoft/codebert-base")  # 处理源代码序列
        # self.classifier = nn.Linear(config.cb_embed, config.num_classes)
        self.prob = nn.Linear(config.cb_embed, config.num_classes)
    

    def forward(self, s, layer, tag=False):
        s_code = s
        codebert_out  = self.codebert(s_code, attention_mask=s_code.ne(1), output_hidden_states=True, output_attentions=True)
        hidden_out = codebert_out[2]
        if tag:
            layer_out = hidden_out[layer][:, 0, :]
            out = self.prob(layer_out)
        else:
            out1 = codebert_out[0][:, 0, :]
            out = self.classifier(out1)
        return out
