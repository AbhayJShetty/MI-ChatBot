import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class vocabGenerate():
    def __init__(self, data):
        self.line_text = ""
        self.data = data

    def lineMerge(self):
        for text in self.data:
            self.line_text += text["input"] + " " + text["output"] + " "
        print(f"Data merge complete!")

    def dataClean(self):
        self.line_text = list(set(self.line_text.lower().split()))
        self.line_text.sort()
        print("Data cleaning complete!")

    def mapWord(self):
        word2idx = {word : idx + 1 for idx, word in enumerate(self.line_text)}
        word2idx['<PAD>'] = 0
        idx2word = {idx : word for word, idx in word2idx.items()}

        vocab_size = len(word2idx)
        print(f"Vocabulary size : {vocab_size}")
        return [word2idx, idx2word, vocab_size]