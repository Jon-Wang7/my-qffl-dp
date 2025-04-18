import torch
from torch.utils.data import Dataset
import numpy as np

class Z_Data(Dataset):
    def __init__(self, data_type, image_type):
        super(Z_Data, self).__init__()
        if data_type == 'train':
            self.data = torch.load('data/mnist/train_data.pkl')
            self.label = torch.load('data/mnist/train_label.pkl')
        elif data_type == 'test':
            self.data = torch.load('data/mnist/test_data.pkl')
            self.label = torch.load('data/mnist/test_label.pkl')
        elif data_type == 'val':
            # 从训练集中划分验证集
            train_data = torch.load('data/mnist/train_data.pkl')
            train_label = torch.load('data/mnist/train_label.pkl')
            val_size = int(0.1 * len(train_data))
            self.data = train_data[:val_size]
            self.label = train_label[:val_size]
            
    def __getitem__(self, index):
        return self.data[index], self.label[index]
        
    def __len__(self):
        return len(self.data) 