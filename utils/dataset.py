import json

import torch
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import get_max_triplet


class Sentence:
    def __init__(self, tokens, relations):
        self.tokens = tokens
        self.mask = []
        self.ids = []
        self.ner_id2span = dict()
        self.ner_span2id = dict()
        self.entity_pair = []
        self.labels = []

        ner_id = 0
        for triplet in relations:
            h_span = (triplet[0], triplet[1])
            t_span = (triplet[2], triplet[3])
            if h_span not in self.ner_id2span:
                self.ner_span2id[h_span] = ner_id
                self.ner_id2span[ner_id] = h_span
                ner_id += 1
            if t_span not in self.ner_id2span:
                self.ner_span2id[t_span] = ner_id
                self.ner_id2span[ner_id] = t_span
                ner_id += 1
            self.entity_pair.append([self.ner_span2id[h_span], self.ner_span2id[t_span]])
            self.labels.append(triplet[4])


class SentenceDataset(Dataset):
    def __init__(self, file_path, rel2id_path, tokenizer):
        rel2id_file = open(rel2id_path, encoding="utf-8")
        self.rel2id = json.load(rel2id_file)
        rel2id_file.close()

        self.data = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                cur_sen = json.loads(line)
                cur_sen = Sentence(cur_sen["tokens"], cur_sen["relations"])
                cur_sen.ids, cur_sen.mask = tokenizer.tokenize(cur_sen.tokens)
                for i in range(len(cur_sen.labels)):
                    cur_sen.labels[i] = self.rel2id[cur_sen.labels[i]]
                self.data.append(cur_sen)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class SentenceClassificationDataset(Dataset):
    def __init__(self, file_path, rel2id_path, tokenizer):
        rel2id_file = open(rel2id_path, encoding="utf-8")
        self.rel2id = json.load(rel2id_file)
        rel2id_file.close()

        self.data = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                cur_data = json.loads(line)

                # add marker
                if cur_data["h"]["pos"][0] < cur_data["t"]["pos"][0]:
                    first_pos = cur_data["h"]["pos"]
                    second_pos = cur_data["t"]["pos"]
                else:
                    first_pos = cur_data["t"]["pos"]
                    second_pos = cur_data["h"]["pos"]
                cur_data["token"].insert(second_pos[1] + 1, "</e2>")
                cur_data["token"].insert(second_pos[0], "<e2>")
                cur_data["token"].insert(first_pos[1] + 1, "</e1>")
                cur_data["token"].insert(first_pos[0], "<e1>")

                cur_data["ids"], cur_data["mask"] = tokenizer.tokenize(cur_data["token"])
                cur_data["label"] = self.rel2id[cur_data["relation"]]

                self.data.append(cur_data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class SSLDataset(Dataset):
    def __init__(self, strong_data, weak_data):
        self.strong_data = strong_data
        self.weak_data = weak_data

    def __getitem__(self, idx):
        return self.strong_data.data[idx], self.weak_data.data[idx]

    def __len__(self):
        return len(self.strong_data.data)


def collate_fn(batch):
    max_triplets_num = get_max_triplet(batch)


def collate_fn_semeval(batch):
    ids = []
    masks = []
    label = []
    for item in batch:
        ids.append(item["ids"])
        masks.append(item["mask"])
        label.append(item["label"])
    ids = torch.tensor(ids, dtype=torch.long)
    masks = torch.tensor(masks, dtype=torch.long)
    label = torch.tensor(label, dtype=torch.long)
    return ids, masks, label


def collate_fn_ssl_semeval(batch):
    strong_data, weak_data = zip(*batch)
    strong_ids, strong_masks, strong_label = collate_fn_semeval(strong_data)
    weak_ids, weak_masks, weak_label = collate_fn_semeval(weak_data)
    ssl_ids = torch.cat((strong_ids, weak_ids), dim=0)
    ssl_masks = torch.cat((strong_masks, weak_masks), dim=0)
    return ssl_ids, ssl_masks
