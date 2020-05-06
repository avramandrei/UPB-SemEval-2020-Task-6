import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

#
# class LangModelWithLSTM(nn.Module):
#     def __init__(self, lang_model, vocab_size, emb_size, input_size, lstm_hidden_size, num_layers, hidden_size, tokenizer, fine_tune):
#         super(LangModelWithLSTM, self).__init__()
#         self.fine_tune = fine_tune
#
#         self.lang_model = lang_model
#         self.lang_model.resize_token_embeddings(vocab_size + 2 if fine_tune else vocab_size)
#
#         self.embedding = nn.Embedding(vocab_size + 2 if fine_tune else vocab_size, emb_size, padding_idx=tokenizer.pad_token_id)
#
#         self.lstm = nn.LSTM(input_size, lstm_hidden_size, num_layers, batch_first=True, dropout=0.1, bidirectional=True)
#
#         self.linear1 = nn.Linear(lstm_hidden_size*2, hidden_size)
#         self.dropout1 = nn.Dropout(0.5)
#         self.linear2 = nn.Linear(hidden_size, hidden_size)
#         self.dropout2 = nn.Dropout(0.5)
#         self.linear3 = nn.Linear(hidden_size, 1)
#
#     def forward(self, x, mask, lengths):
#         if self.fine_tune:
#             lang_model_embeddings = self.lang_model(x, attention_mask=mask)[0]
#         else:
#             with torch.no_grad():
#                 self.lang_model.eval()
#                 lang_model_embeddings = self.lang_model(x, attention_mask=mask)[0]
#
#         embeddings = torch.cat((lang_model_embeddings, self.embedding(x)), dim=2)
#
#         embeddings = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
#
#         lstm_output, _ = self.lstm(embeddings)
#
#         lstm_output = gelu(pad_packed_sequence(lstm_output, batch_first=True)[0][:, -1, :])
#
#         output = self.dropout1(gelu(self.linear1(lstm_output)))
#         output = self.dropout2(gelu(self.linear2(output)))
#         output = F.sigmoid(self.linear3(output))
#
#         return output


class LangModelWithDense(nn.Module):
    def __init__(self, lang_model, vocab_size, input_size, hidden_size, fine_tune):
        super(LangModelWithDense, self).__init__()
        self.lang_model = lang_model
        self.lang_model.resize_token_embeddings(vocab_size + 2 if fine_tune else vocab_size)

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.8 if fine_tune else 0.1)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.8 if fine_tune else 0.1)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.fine_tune = fine_tune

    def forward(self, x, mask):
        if self.fine_tune:
            embeddings = self.lang_model(x, attention_mask=mask)[0]
        else:
            with torch.no_grad():
                self.lang_model.eval()
                embeddings = self.lang_model(x, attention_mask=mask)[0]

        if "xlnet" in str(type(self.lang_model)):
            embeddings = embeddings[:, 0, :]
        else:
            embeddings = torch.mean(embeddings, dim=1)

        output = self.dropout1(F.gelu(self.linear1(embeddings)))
        output = self.dropout2(F.gelu(self.linear2(output)))
        output = F.sigmoid(self.linear3(output))

        return output