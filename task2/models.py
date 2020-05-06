import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LangModelWithLSTM(nn.Module):
    def __init__(self, lang_model, vocab_size, emb_size, input_size, lstm_hidden_size, num_layers,
                 hidden_size, num_classes, tokenizer, fine_tune):

        super(LangModelWithLSTM, self).__init__()
        self.num_classes = num_classes
        self.fine_tune = fine_tune

        self.lang_model = lang_model

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=tokenizer.pad_token_id)
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, num_layers, batch_first=True, dropout=0.2, bidirectional=True)

        self.linear1 = nn.Linear(lstm_hidden_size*2, hidden_size)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, mask, lengths):
        if not self.fine_tune:
            with torch.no_grad():
                self.lang_model.eval()
                lang_model_embeddings = self.lang_model(x, attention_mask=mask)[0]
        else:
            lang_model_embeddings = self.lang_model(x, attention_mask=mask)[0]

        embeddings = torch.cat((lang_model_embeddings, self.embedding(x)), dim=2)

        embeddings = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)

        lstm_output = self.lstm(embeddings)[0]

        lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True)

        batch_size = lstm_output.shape[0]
        seq_len = lstm_output.shape[1]

        outputs = torch.zeros((batch_size, seq_len, self.num_classes))
        for i in range(seq_len):
            output = self.dropout1(F.leaky_relu(self.linear1(lstm_output[:, i, :])))
            output = self.dropout2(F.leaky_relu(self.linear2(output)))
            output = self.linear3(output)

            outputs[:, i, :] = output

        return outputs


class LangModelWithDense(nn.Module):
    def __init__(self, lang_model, input_size, hidden_size, num_classes, fine_tune):

        super(LangModelWithDense, self).__init__()
        self.num_classes = num_classes
        self.fine_tune = fine_tune

        self.lang_model = lang_model

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.8 if fine_tune else 0.1)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.8 if fine_tune else 0.1)
        self.linear3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, mask):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        if not self.fine_tune:
            with torch.no_grad():
                self.lang_model.eval()
                embeddings = self.lang_model(x, attention_mask=mask)[0]
        else:
            embeddings = self.lang_model(x, attention_mask=mask)[0]

        outputs = torch.zeros((batch_size, seq_len, self.num_classes))
        for i in range(seq_len):
            output = self.dropout1(F.gelu(self.linear1(embeddings[:, i, :])))
            output = self.dropout2(F.gelu(self.linear2(output)))
            output = self.linear3(output)

            outputs[:, i, :] = output

        return outputs