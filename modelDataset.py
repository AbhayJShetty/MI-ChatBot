import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class modelDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        line = self.data[index]
        return torch.tensor(line["input_idx"]), torch.tensor(line["output_idx"])