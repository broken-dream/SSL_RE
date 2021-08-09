import json
import random


class WeakAugment:
    def __init__(self, dictionary):
        self.aug_num = 4
        self.dictionary = dictionary
        self.aug_map = {
            0: self.del_token,
            1: self.add_token,
            2: self.replace_token,
            3: self.swap_token
        }

    @staticmethod
    def gen_idx(length, ban_idx):
        idx = random.randint(0, length-1)
        while idx in ban_idx:
            idx = random.randint(0, length-1)
        return idx

    def augment(self, tokens, relations):
        ban_idx = []
        for item in relations:
            h_start = item[0]
            h_end = item[1]
            t_start = item[2]
            t_end = item[3]
            ban_idx += range(h_start, h_end+1)
            ban_idx += range(t_start, t_end+1)

        aug_function = self.aug_map[random.randint(0, 3)]
        print(aug_function.__name__)
        aug_function(tokens, relations, ban_idx)

    @staticmethod
    def del_token(tokens, relations, ban_idx):
        del_idx = WeakAugment.gen_idx(len(tokens), ban_idx)

        tokens.pop(del_idx)
        for triplet in relations:
            for i in range(4):
                if triplet[i] > del_idx:
                    triplet[i] -= 1

    def add_token(self, tokens, relations, ban_idx):
        add_idx = WeakAugment.gen_idx(len(tokens), ban_idx)

        added_token = random.sample(self.dictionary, 1)
        tokens.insert(add_idx, added_token[0])
        for triplet in relations:
            for i in range(4):
                if triplet[i] > add_idx:
                    triplet[i] += 1

    def replace_token(self, tokens, relations,  ban_idx):
        replace_idx = WeakAugment.gen_idx(len(tokens), ban_idx)
        replaced_token = random.sample(self.dictionary, 1)
        while replaced_token[0] == tokens[replace_idx]:
            replaced_token = random.sample(self.dictionary, 1)
        tokens[replace_idx] = replaced_token[0]

    @staticmethod
    def swap_token(tokens, relations, ban_idx):
        t1_pos = WeakAugment.gen_idx(len(tokens), ban_idx)
        t2_pos = WeakAugment.gen_idx(len(tokens), ban_idx)
        while t2_pos == t1_pos or tokens[t1_pos] == tokens[t2_pos]:
            t2_pos = WeakAugment.gen_idx(len(tokens), ban_idx)
        tokens[t1_pos], tokens[t2_pos] = tokens[t2_pos], tokens[t1_pos]


def weak_aug_semeval(input_path, out_path, augment: WeakAugment):
    out_file = open(out_path, "w+", encoding="utf-8")
    cnt = 0
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            print(cnt)
            cnt += 1
            cur_data = json.loads(line)
            relations = []
            triplet = cur_data["h"]["pos"] + cur_data["t"]["pos"]
            relations.append(triplet)
            augment.augment(cur_data["token"], relations)
            cur_data["h"]["pos"] = [relations[0][0], relations[0][1]]
            cur_data["t"]["pos"] = [relations[0][2], relations[0][3]]
            print(json.dumps(cur_data), file=out_file)
    out_file.close()


if __name__ == "__main__":
    dic_file = open("../data/dictionary.json", encoding="utf-8")
    dic = json.load(dic_file)
    dic_file.close()
    aug = WeakAugment(dic)
    weak_aug_semeval("../data/processed_semeval/train.json", "../data/weak_semeval/train.json", aug)
    # weak_aug_semeval("../data/processed_semeval/test.json", "../data/weak_semeval/test.json", aug)
    # weak_aug_semeval("../data/processed_semeval/val.json", "../data/weak_semeval/val.json", aug)
