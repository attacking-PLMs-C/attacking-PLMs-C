from turtle import forward
import numpy as np
import math
import torch.nn.functional as F
from unicodedata import bidirectional
import torch
from scipy.stats import entropy
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn


class Config:
    '''模型参数配置'''
    def __init__(self):
        self.model_name = 'CodeBERT'
        self.save_path = './save_dict/' + self.model_name + '.ckpt'  # 保存模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

        self.num_classes = 2  #类别数
        self.num_epochs = 2  
        self.max_repoch = 2
        self.batch_size = 16  #mini_batch大小
        self.pad_size = 512  #每句话处理的长度大小（截取或填补）
        self.learning_rate = 5e-5  #学习率 
        self.cb_embed = 768
        self.hidden_size = 256  
    

class CodeBERT(nn.Module):
    def __init__(self, config):
        super(CodeBERT, self).__init__()
        self.codebert = RobertaModel.from_pretrained("microsoft/codebert-base")  # 处理源代码序列
        self.classifier = nn.Sequential(
                                    nn.Linear(config.cb_embed, config.num_classes)
                                    )
    

    def forward(self, s):
        s_code = s
        codebert_out  = self.codebert(s_code, attention_mask=s_code.ne(1), output_hidden_states=True, output_attentions=True)
        last_out = codebert_out[0][:, 0, :]
        # print(type(last_out))
        hidden_out = codebert_out[2]
        attention_out = codebert_out[3]  #[layer_num, batch_size, head_num, sequence_length, sequence_length]

        out = self.classifier(last_out)
        # print(type(out))
        
        return out