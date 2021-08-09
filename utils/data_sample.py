import json
import random


def sample(input_path, out_path, sample_num):
    input_file = open(input_path, encoding="utf-8")
    data = json.load(input_file)
    input_file.close()
    sampled_data = random.sample(data, sample_num)
    out_file = open(out_path, "w+", encoding="utf-8")
    print(json.dumps(sampled_data), file=out_file)
    out_file.close()