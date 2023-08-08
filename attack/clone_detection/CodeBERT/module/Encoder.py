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
        self.model_name = 'Encoder'
        self.save_path = './save_dict/' + self.model_name + '_3.ckpt'  # 保存模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

        self.num_classes = 2  #类别数
        self.num_epochs = 5  
        self.max_repoch = 2
        self.batch_size = 16  #mini_batch大小
        self.pad_size = 512  #每句话处理的长度大小（截取或填补）
        self.learning_rate = 2e-5  #学习率 
        self.cb_embed = 768
        self.hidden_size = 256  
    

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.codebert = RobertaModel.from_pretrained("microsoft/codebert-base")  # 处理源代码序列
        self.classifier = nn.Sequential(
                                    nn.Linear(config.cb_embed,  config.num_classes)
                                    )
    

    def forward(self, s):
        s_code = s
        s_code = s_code.unsqueeze(0)
        # t_code = t
        atten_mask1 = torch.ones(s_code.size(), dtype=torch.long).cuda()
        atten_mask1[s_code == 1] = 0
        codebert_out1  = self.codebert(s_code, attention_mask=atten_mask1, output_hidden_states=True, output_attentions=True)
        # atten_mask2 = torch.ones(t_code.size(), dtype=torch.long).cuda()
        # atten_mask2[t_code == 1] = 0
        # codebert_out2  = self.codebert(t_code, attention_mask=atten_mask2, output_hidden_states=True, output_attentions=True)
        last_out = codebert_out1[0][:, 0, :]
        hidden_out1 = codebert_out1[2]
        # hidden_out2 = codebert_out2[2]
        attention_out1 = codebert_out1[3]  #[layer_num, batch_size, head_num, sequence_length, sequence_length]
        # attention_out2 = codebert_out2[3]

        # attention_out1_6_12 = attention_out1[5:]
        # hidden_out1_6_12 = hidden_out1[6:]
        # attention_out2_6_12 = attention_out2[5:]
        # hidden_out2_6_12 = hidden_out2[6:]
        # layer_out = hidden_out[la][:, 0, :]
        hidden_out1 = torch.cat([torch.unsqueeze(n, 0) for n in hidden_out1], dim=0)
        hidden_out1 = hidden_out1.transpose(0,1)
        attention_out1 = torch.cat([torch.unsqueeze(n, 0) for n in attention_out1], dim=0)
        attention_out1 = attention_out1.transpose(0,1)
        out = self.classifier(last_out)
        # print(type(out))
        
        return out, hidden_out1, attention_out1