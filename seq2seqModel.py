import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden, dec_hidden):
        super().__init__()
        self.W1 = nn.Linear(enc_hidden * 2, dec_hidden)
        self.W2 = nn.Linear(dec_hidden, dec_hidden)
        self.V = nn.Linear(dec_hidden, 1)

    def forward(self, encoder_outputs, hidden):
        hidden = hidden.unsqueeze(1)
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(hidden)))
        attn_weights = F.softmax(score, dim=1)
        context = (attn_weights * encoder_outputs).sum(dim=1)
        return context, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_size=128, hidden_size=256, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.encoder = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        self.attn = BahdanauAttention(hidden_size, hidden_size)
        self.enc_to_dec = nn.Linear(hidden_size * 2, hidden_size)
        self.decoder = nn.LSTM(emb_size + hidden_size * 2, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.size()
        outputs = []

        enc_emb = self.dropout(self.embedding(src))
        enc_out, (h, c) = self.encoder(enc_emb)

        dec_h = torch.tanh(self.enc_to_dec(torch.cat((h[-2], h[-1]), dim=1))).unsqueeze(0)
        dec_c = torch.tanh(self.enc_to_dec(torch.cat((c[-2], c[-1]), dim=1))).unsqueeze(0)

        dec_input = trg[:, 0]

        for t in range(1, trg_len):
            emb_dec_in = self.dropout(self.embedding(dec_input))
            context, _ = self.attn(enc_out, dec_h[-1])
            lstm_input = torch.cat((emb_dec_in, context), dim=1).unsqueeze(1)

            out, (dec_h, dec_c) = self.decoder(lstm_input, (dec_h, dec_c))
            logits = self.fc(out.squeeze(1))

            outputs.append(logits.unsqueeze(1))

            top1 = logits.argmax(1)
            use_tf = torch.rand(1).item() < teacher_forcing_ratio
            dec_input = trg[:, t] if use_tf else top1

        outputs = torch.cat(outputs, dim=1)
        return outputs
