import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMatch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.labeled_memory = torch.tensor(config.num_labeled, config.hidden_size)
        self.labeled_logits = torch.tensor(config.num_labeled, config.class_num)

    def cal_sim(self, data):
        # data shape: batch*hidden_size
        ids = F.cosine_similarity(data.unsqueeze(1), self.labeled_memory.unsqueeze(0), dim=-1).argmax(dim=1) # batch*num_labeled
        batch_logits = torch.index_select(self.labeled_logits, dim=0, index=ids) # batch*class_num
        return batch_logits

    def update_logits(self, weak_data, weak_logits):
        batch_logits = self.cal_sim(weak_data)
        res_logits = batch_logits*self.config.labeled_weight + (1-self.config.num_labeled)*weak_logits
        return res_logits

    def update_memory(self, labled_idx, data, logits):
        self.labeled_memory.index_copy_(0, labled_idx, data)
        self.labeled_logits.index_copy_(0, labled_idx, logits)