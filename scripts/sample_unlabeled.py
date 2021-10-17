import json
import random

def get_data(input_path):
    res = []
    with open(input_path) as f:
        for line in f:
            res.append(json.loads(line))
    return res

def print_data(data, out_path):
    out_file = open(out_path, "w+", encoding="utf-8")
    for item in data:
        print(json.dumps(item), file=out_file)
    out_file.close()

def get_train_idx(input_path):
    f = open(input_path)
    data = json.load(f)
    f.close()
    return data["train_idx"]

if __name__ == "__main__":
    init_data = get_data("../data/processed_semeval/val.json")
    weak_data = get_data("../data/weak_semeval/val.json")
    strong_data = get_data("../data/strong_semeval/val.json")
    all_idx = range(len(init_data))
    train_idx = get_train_idx("")
    for idx in train_idx:
        del all_idx[idx]
    
    unlabeled_num = 6200
    unlabeled_idx = random.sample(all_idx, unlabeled_num)
    weak_train = []
    strong_train = []
    for idx in unlabeled_idx:
        weak_train.append(weak_data[idx])
        strong_train.append(strong_data[idx])
    
    
    print_data(weak_train, "../data/nero_processed_semeval/weak_append.json")
    print_data(strong_train, "../data/nero_processed_semeval/strong_append.json")

