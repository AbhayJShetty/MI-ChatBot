import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class seq2seqModel(nn.Module):
    def __init__(self, vocab_size, emb_size = 64, hidden_state = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.encoder = nn.LSTM(emb_size, hidden_state, batch_first = True, bidirectional = True)
        self.decoder = nn.LSTM(hidden_state * 2, hidden_state * 2, batch_first = True)
        self.attn = nn.Linear(emb_size + hidden_state * 2, hidden_state * 2)
        self.fc = nn.Linear(hidden_state * 2, vocab_size)

    def forward(self, encrypt, decrypt):
        encrypt_emb = self.embedding(encrypt)
        encrypt_out, (hidden, cell) = self.encoder(encrypt_emb)

        decrypt_emb = self.embedding(decrypt)
        batch_size, decrypt_len, _ = decrypt_emb.size()

        output = []
        dec_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1).unsqueeze(0)
        dec_cell = torch.cat((cell[-2], cell[-1]), dim=1).unsqueeze(0)

        for t in range(decrypt_len):
            dec_t = decrypt_emb[:, t, :].unsqueeze(1)

            attn_weights = torch.bmm(dec_hidden.permute(1,0,2), encrypt_out.permute(0,2,1))
            attn_weights = F.softmax(attn_weights, dim=2)

            context = torch.bmm(attn_weights, encrypt_out)

            dec_input_combined = torch.cat((dec_t, context), dim=2)
            dec_input_combined = self.attn(dec_input_combined)

            dec_output, (dec_hidden, dec_cell) = self.decoder(dec_input_combined, (dec_hidden, dec_cell))
            out = self.fc(dec_output.squeeze(1))
            output.append(out.unsqueeze(1))

        output = torch.cat(output, dim=1)
        return output