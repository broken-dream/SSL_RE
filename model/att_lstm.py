import torch
import torch.nn as nn
import torch.nn.functional as F


class AttLSTM(nn.Module):
    def __init__(self, encoder, config):
        super().__init__()
        self.encoder = encoder
        self.class_num = config.class_num

        self.linear_dropout = nn.Dropout(config.linear_dropout_value)
        self.linear = nn.Linear(
            in_features=self.encoder.hidden_size,
            out_features=self.class_num,
            # bias=True
            bias=False
        )
        self.tanh = nn.Tanh()
        self.att_weight = torch.nn.Parameter(torch.randn(1, self.encoder.hidden_size, 1))

    def attention_layer(self, x, mask):
        att_weight = self.att_weight.expand(x.shape[0], -1, -1)  # B*H*1
        att_score = torch.bmm(self.tanh(x), att_weight)  # B*L*H * B*H*1 -> B*L*1

        mask = mask.unsqueeze(dim=-1)  # B*L*1
        att_score = att_score.masked_fill(mask.eq(0), float('-inf'))
        att_weight = F.softmax(att_score, dim=1)

        res = torch.bmm(x.transpose(1, 2), att_weight).squeeze(dim=-1)  # B*H*L * B*L*1 -> B*H*1 -> B*H
        res = self.tanh(res)
        return res

    def forward(self, ids, mask):
        x = self.encoder(ids)  # B*L*H
        x = self.attention_layer(x, mask)
        x = self.linear_dropout(x)
        logits = self.linear(x)
        return logits
