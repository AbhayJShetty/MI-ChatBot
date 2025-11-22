from fileReader import fileReader
from vocabGenerate import vocabGenerate
from tokenizeData import tokenizeData
from modelDataset import modelDataset
from seq2seqModel import seq2seqModel
from modelTrain import modelTrain
from modelInference import modelInference

import torch
from torch.utils.data import DataLoader

class mainChatbot():
    def __init__(self):
        #inputs = input("Enter your symptoms : ")
        inputs = "I have headache. My body has hot. I cough a lot."
        fl = fileReader()
        extracted_data = fl.readFile()

        vg = vocabGenerate(extracted_data)
        vg.lineMerge()
        vg.dataClean()
        word2idx, idx2word, vocab_size = vg.mapWord()

        td = tokenizeData(extracted_data, word2idx)
        tokenized_data, maxInput = td.generateTokens()

        dataset = modelDataset(tokenized_data)
        dataloader = DataLoader(dataset, batch_size = 10, shuffle = True)

        print(f"Data is loaded and batched!")

        torch.manual_seed(42)
        model = seq2seqModel(vocab_size)
        print(f"Model Created!")

        trainer = modelTrain(model, dataloader, word2idx)
        trainer.train()

        inference = modelInference(model, inputs, word2idx, idx2word, maxInput)
        response = inference.generateOutput()

        print(f"**The below instructions may or may not verified. Please cross check with a verified doctor!**")
        print(f"Response : {response}")
    
if __name__ == "__main__":
    mainChatbot()