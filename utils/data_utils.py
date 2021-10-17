import random
import numpy as np
import torch
from tqdm import tqdm

def get_max_triplet(batch):
    # get instance with most triplets in current batch
    max_count = 0
    for sen in batch:
        if len(sen.triplets) > max_count:
            max_count = len(sen.triplets)
    return max_count

def gen_training_examples(batch, max_triplets_num):
    # padding triplets and gen negative examples
    res = []
    for sen in batch:
        # the original data shouldn't be changed
        triplets_copy = []
        for item in sen.triplets:
            triplets_copy.append(item)

        neg_count = max_triplets_num - len(sen.entity_pair)

        neg_entity_pair = []
        # generate strong negative examples, i.e.
        # pairs of golden entities that are not related
        for i in range(len(sen.ner_id2pos)):
            for j in range(len(sen.ner_id2pos)):
                if i != j and (i, j) not in sen.entity_pair:
                    neg_entity_pair.append((i, j))

        if len(neg_entity_pair) > neg_count:
            sen.entity_pair += random.sample(neg_entity_pair, neg_count)
        else:
            for i in range(neg_count - len(neg_entity_pair)):
                pass


def get_word_vec_semeval(data_path, dim):
    data = []
    word2id = dict()

    word2id["[PAD]"] = len(word2id)
    word2id["[UNK]"] = len(word2id)
    word2id["<e1>"] = len(word2id)
    word2id["</e1>"] = len(word2id)
    word2id["<e2>"] = len(word2id)
    word2id["</e2>"] = len(word2id)
    idx = len(word2id)
    with open(data_path, encoding="utf-8") as f:
        for line in tqdm(f):
            cur_emb = line.split()
            word = "".join(cur_emb[0:-dim])
            word2id[word] = idx
            data.append(list(map(float,cur_emb[-dim:])))
            idx += 1
            if idx > 600000:
                break
    word_emb = np.array(data)

    print(word_emb.shape)
    pad_emb = np.zeros((1, word_emb.shape[-1]))
    marker_emb = word_emb.mean(axis=0)
    marker_emb = np.expand_dims(marker_emb, axis=0)
    marker_emb = np.repeat(marker_emb, repeats=5, axis=0)

    word_emb = np.concatenate((pad_emb, marker_emb, word_emb), axis=0)
    return word2id, torch.from_numpy(word_emb)



if __name__ == "__main__":
    get_word_vec_semeval("/home/wh/pretrain/glove.6B.300d/glove.6B.300d.npy")