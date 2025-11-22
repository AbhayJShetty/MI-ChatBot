import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class modelTrain():
    def __init__(self, model, data, word2idx):
        self.model = model
        self.data = data
        self.word2idx = word2idx
        self.loss_fn = nn.CrossEntropyLoss(ignore_index = 0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode = "min",
            factor = .5,
            patience = 3,
        )

    def train(self):
        print("Begin model training!")
        epochs = 3
        vocab_size = len(self.word2idx)

        for epoch in range(epochs):
            total_loss = 0

            for batch, (enc_input, dec_output) in enumerate(self.data):
                dec_input = torch.cat([torch.tensor([[self.word2idx['<PAD>']]]*enc_input.size(0)), dec_output[:, :-1]], dim=1)
                self.optimizer.zero_grad()
                output = self.model(enc_input, dec_input)
                loss = self.loss_fn(output.view(-1, vocab_size), dec_output.view(-1))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                print(f"Batch : {batch + 1} | Loss : {loss:.4f}")

            self.scheduler.step(total_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1} | Loss: {total_loss/len(self.data):.4f} | lr : {current_lr}")
        
        print(f"Ended model training!")