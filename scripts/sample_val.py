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

if __name__ == "__main__":
    init_val = get_data("../data/processed_semeval/val.json")
    weak_val = get_data("../data/weak_semeval/val.json")
    strong_val = get_data("../data/strong_semeval/val.json")
    all_idx = range(len(init_val))
    val2train_idx = random.sample(all_idx, len(init_val)-800)
    new_val = []
    weak_train = []
    strong_train = []
    for idx in range(len(init_val)):
        if idx in val2train_idx:
            weak_train.append(weak_val[idx])
            strong_train.append(strong_val[idx])
        else:
            new_val.append(init_val[idx])
    
    print_data(new_val, "../data/nero_processed_semeval/val.json")
    print_data(weak_train, "../data/nero_processed_semeval/weak_append.json")
    print_data(strong_train, "../data/nero_processed_semeval/strong_append.json")

