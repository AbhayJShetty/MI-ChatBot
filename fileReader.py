import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class fileReader():
    def __init__(self):
        self.limit = 100
        self.file = r"D:\chatBot\medical_data\HealthCareMagic-100k.json"

    def readFile(self):
        with open(self.file, "r") as f:
            extract = json.load(f) 
        data = [{"input" : query["input"], "output" : query["output"]} for query in extract[:]]

        print(f"Loaded {len(data)} records!")
        return data