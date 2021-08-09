import json

bracket_map = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]"
}


def reformat_scierc_bracket(doc):
    for sen in doc["sentences"]:
        for i in range(len(sen)):
            if sen[i] in bracket_map:
                sen[i] = bracket_map[sen[i]]


# change the global index to local index
def reformat_scierc_token_idx(doc):
    token_num = 0
    for idx in range(len(doc["sentences"])):
        for ner in doc["ner"][idx]:
            ner[0] -= token_num
            ner[1] -= token_num
        for triplet in doc["relations"][idx]:
            triplet[0] -= token_num
            triplet[1] -= token_num
            triplet[2] -= token_num
            triplet[3] -= token_num
        token_num += len(doc["sentences"][idx])


def divide_scierc(doc):
    res = []
    for i in range(len(doc["sentences"])):
        if len(doc["relations"][i]) == 0:
            continue
        new_doc = dict()
        new_doc["tokens"] = doc["sentences"][i]
        new_doc["ner"] = doc["ner"][i]
        new_doc["relations"] = doc["relations"][i]
        res.append(new_doc)
    return res


def reformat_scierc(input_path, out_path):
    out_file = open(out_path, "w+", encoding="utf-8")
    res = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            cur_doc = json.loads(line)
            reformat_scierc_bracket(cur_doc)
            reformat_scierc_token_idx(cur_doc)
            res += divide_scierc(cur_doc)
    for item in res:
        print(json.dumps(item), file=out_file)
    out_file.close()


def reformat_semeval(input_path, out_path):
    out_file = open(out_path, "w+", encoding="utf-8")
    res = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            cur_doc = json.loads(line)
            cur_doc["h"]["pos"][1] -= 1
            cur_doc["t"]["pos"][1] -= 1
            res.append(cur_doc)
    for item in res:
        print(json.dumps(item), file=out_file)
    out_file.close()

if __name__ == "__main__":
    # reformat_scierc("../data/scierc/train.json", "../data/scierc_sentence/train.json")
    reformat_scierc("../data/scierc/test.json", "../data/scierc_sentence/test.json")
    # reformat_scierc("../data/scierc/dev.json", "../data/scierc_sentence/dev.json")

    # reformat_semeval("../data/semeval/semeval_train.json", "../data/processed_semeval/train.json")
    # reformat_semeval("../data/semeval/semeval_test.json", "../data/processed_semeval/test.json")
    # reformat_semeval("../data/semeval/semeval_val.json", "../data/processed_semeval/val.json")