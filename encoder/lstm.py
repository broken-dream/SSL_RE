import torch
import torch.nn as nn


class BiLSTMEncoder(nn.Module):
    def __init__(self, embedding, config, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=embedding,
            freeze=False,
            padding_idx=padding_idx
        )

        self.max_len = config.max_len
        self.word_emb_dim = config.word_emb_dim
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.emb_dropout_value = config.emb_dropout_value
        self.lstm_dropout_value = config.lstm_dropout_value

        self.lstm = nn.LSTM(
            input_size=self.word_emb_dim,
            hidden_size=self.hidden_size//2,
            num_layers=self.num_layers,
            bias=True,
            bidirectional=True
        )

        self.emb_dropout = nn.Dropout(self.emb_dropout_value)
        self.lstm_dropout = nn.Dropout(self.lstm_dropout_value)

    def forward(self, data):
        x = self.embedding(data).float()  # B*L*emb_size
        x = self.emb_dropout(x)
        x = x.transpose(0, 1)
        x, h = self.lstm(x)
        x = self.lstm_dropout(x)  # L*B*H
        return x.transpose(1, 0)  # B*L*H



