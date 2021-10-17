import json
import numpy as np

dic_file = open("/home/wh/pretrain/glove.6B.300d/word2id.6B.json", encoding="utf-8")
dic = json.load(dic_file)
dic_file.close()

data = np.load("/home/wh/pretrain/glove.6B.300d/glove.6B.300d.npy")
data = data.tolist()

out_file = open("/home/wh/pretrain/glove.6B.300d/glove.6B.300d.txt", "w+", encoding="utf-8")
for token,idx in dic.items():
    if token != "[PAD]" and token != "[UNK]":
        res = token
        for entry in data[idx]:
            res += " " + str(entry)
        print(res, file=out_file)
out_file.close()