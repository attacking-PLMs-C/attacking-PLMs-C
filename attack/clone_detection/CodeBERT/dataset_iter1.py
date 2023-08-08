import pandas as pd
import torch

class DatasetIterdtor:
    '''生成可迭代数据集'''
    def __init__(self, batches, batch_size, device):
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.batch_size = batch_size
        self.device = device
        self.residue = False  # 记录batch数量是否为正数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0

    def _to_tensor(self, data):
        id_ = [item[0] for item in data]
        s_code = torch.tensor([item[1] for item in data]).to(self.device)
        initial_code = torch.tensor([item[2] for item in data]).to(self.device)
        m2_code = torch.tensor([item[3] for item in data]).to(self.device)
        m5_code = torch.tensor([item[4] for item in data]).to(self.device)
        m7_code = torch.tensor([item[5] for item in data]).to(self.device)
        label = torch.LongTensor([int(item[6]) for item in data]).to(self.device)
        return id_, s_code, initial_code, m2_code, m5_code, m7_code, label

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration

        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
    
    def __iter__(self):
        return self
    
    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches