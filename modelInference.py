import torch
import torch.nn.functional as F
import tokenizeData as T

class modelInference():
    def __init__(self, model, input, word2idx, idx2word, maxInput):
        self.model = model
        self.input = input
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.maxInput = maxInput
        self.maxLen = 20

    def generateOutput(self):
        self.model.eval()

        input_ids = T.tokenizeData([], self.word2idx).tokenize(self.input)
        input_ids += [0] * (self.maxInput - len(input_ids))
        enc_input = torch.tensor([input_ids])

        dec_input = torch.tensor([[self.word2idx['<PAD>']]])

        enc_emb = self.model.embedding(enc_input)
        enc_out, (hidden, cell) = self.model.encoder(enc_emb)

        dec_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1).unsqueeze(0)
        dec_cell = torch.cat((cell[-2], cell[-1]), dim=1).unsqueeze(0)

        response = []

        for _ in range(self.maxLen):
            dec_emb = self.model.embedding(dec_input[:, -1:].contiguous())

            attn_w = torch.bmm(
                dec_hidden.permute(1, 0, 2),
                enc_out.permute(0, 2, 1)
            )
            attn_w = F.softmax(attn_w, dim=2)

            context = torch.bmm(attn_w, enc_out)

            combined = torch.cat((dec_emb, context), dim=2)
            combined = self.model.attn(combined)

            dec_out, (dec_hidden, dec_cell) = self.model.decoder(
                combined, (dec_hidden, dec_cell)
            )

            logits = self.model.fc(dec_out[:, -1, :])
            pred_id = torch.argmax(logits, dim=1).item()

            if pred_id == 0:
                break

            response.append(self.idx2word[pred_id])

            dec_input = torch.cat(
                [dec_input, torch.tensor([[pred_id]])], dim=1
            )

        return " ".join(response)