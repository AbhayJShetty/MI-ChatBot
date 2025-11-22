import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class tokenizeData():
    def __init__(self, data, word2idx):
        self.data = data
        self.word2idx = word2idx

    def tokenize(self, text):
        return [self.word2idx[word] for word in text.lower().split()]
    
    def generateTokens(self):
        maxInput = max(len(line["input"].split()) for line in self.data)
        maxOutput = max(len(line["output"].split()) for line in self.data)

        for line in self.data:
            line["input_idx"] = self.tokenize(line["input"]) + [0] * (maxInput - len(line["input"].split()))
            line["output_idx"] = self.tokenize(line["output"]) + [0] * (maxOutput - len(line["output"].split()))
        
        print(f"Complete tokenizing!")
        return self.data, maxInput