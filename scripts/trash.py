import json

def get_data(data_path):
    res = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            res.append(json.loads(line))
    
    return res

init_val= get_data("../data/processed_semeval/val.json")
appended_val = get_data("../data/nero_processed_semeval/val.json")
res = []
for item in init_val:
    if item not in appended_val:
        res.append(item)
        #print(item)


out_file = open("../data/append_unlabeled.json", "w+", encoding="utf-8")
for item in res:
    #print(item)
    print(json.dumps(item), file=out_file)
out_file.close()