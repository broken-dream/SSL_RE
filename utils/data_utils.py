import random
import numpy as np
import torch


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


def get_word_vec_semeval(data_path):
    word_vec = np.load(data_path)
    marker_emb = word_vec.mean()
    marker_emb = np.expand_dims(marker_emb, axis=0)
    marker_emb = np.repeat(marker_emb, repeats=4, axis=0)
    word_vec = np.concatenate((word_vec, marker_emb), axis=0)
    return torch.from_numpy(word_vec)
