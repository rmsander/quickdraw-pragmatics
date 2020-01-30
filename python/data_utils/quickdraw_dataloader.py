import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset


import os

import numpy as np

# Now let's use this PointNet class
class QuickDrawDataset(Dataset):
    def __init__(self, data, labels):

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.N)

    def __getitem__(self, index):
        return data[index], label[index]