# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

class pGesture_Dataset(Dataset):
    def __init__(self, annotations_file, file_dir, transform=None, target_transform=None):
        self.labels = pd.read_csv(annotations_file)
        self.file_dir = file_dir
        
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = os.path.join(self.file_dir, self.labels.iloc[idx, 0])
        
#        dataset = np.loadtxt(path, delimiter = ",", skiprows = 1)
        dataset = np.loadtxt(path, delimiter = ",")
        label = self.labels.iloc[idx, 1]
        
        if self.transform:
            dataset = self.transform(dataset)
        if self.target_transform:
            label = self.target_transform(label)
            
        sample = {"dataset": dataset, "label": label, "name": self.labels.iloc[idx, 0]}
        return sample
    