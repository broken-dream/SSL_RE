import json

from transformers import BertTokenizer


class BertSentenceTokenizer:
    def __init__(self, pretrain_path, max_len, pad_token="[PAD]"):
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.max_len = max_len
        self.pad_token = pad_token

    def tokenize(self, input_tokens):
        # add "[CLS]", "[SEP]"
        input_tokens = ["[CLS]"] + input_tokens + ["SEP"]
        init_length = len(input_tokens)

        # padding
        input_tokens = input_tokens + [self.pad_token] * (self.max_len - init_length)
        attention_mask = [1] * init_length + [0] * (self.max_len - init_length)

        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        return input_ids, attention_mask


class GloveSentenceTokenizer:
    def __init__(self, word2id, max_len, pad_token="[PAD]", unk_token="[UNK]", cascade=False):
        self.vocab = word2id
        self.max_len = max_len
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.cascade = cascade

    def tokenize(self, input_tokens):
        input_ids = []
        for token in input_tokens:
            if not self.cascade:
                cur_token = token.lower()
            else:
                cur_token = token
            if cur_token in self.vocab:
                input_ids.append(self.vocab[cur_token])
            else:
                input_ids.append(self.vocab[self.unk_token])

        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]

        attention_mask = [1]*len(input_ids)

        attention_mask += [0] * (self.max_len - len(input_ids))
        input_ids += [self.vocab[self.pad_token]] * (self.max_len - len(input_ids))

        return input_ids, attention_mask
