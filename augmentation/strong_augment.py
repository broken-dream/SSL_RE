import json
import nltk
from augmentation.translation import translate
import time
import tqdm

error_ids = []

def tokens2str(tokens):
    res = ""
    for item in tokens:
        res += item + " "
    return res


def check_nested(ner_pos_list):
    for i in range(len(ner_pos_list)):
        nest_flag = False
        for j in range(len(ner_pos_list)):
            if i == j:
                continue
            if ner_pos_list[i][0] >= ner_pos_list[j][0] and ner_pos_list[i][1] <= ner_pos_list[j][1]:
                ner_pos_list[i].append(j)
                nest_flag = True
                break
        if not nest_flag:
            ner_pos_list[i].append(-1)


def replace_ent(tokens, ner_pos_list):
    ent_marker = "E{}"
    for i in range(len(ner_pos_list)-1,  -1, -1):
        if ner_pos_list[i][3] == -1:
            h_pos = ner_pos_list[i][0]
            t_pos = ner_pos_list[i][1]
            tokens[h_pos] = ent_marker.format(i)
            del tokens[h_pos+1: t_pos+1]
            marker_idx = tokens.index(ent_marker.format(i))
            tokens.insert(marker_idx, "$")
            tokens.insert(marker_idx+2, "$")


def replace_ent_marker(tokens, ner_tokens, ner_pos_list):
    error_flag = False
    while "$" in tokens:
        tokens.remove("$")
    ent_marker_list = ["E{}", "E{}s", "e{}", "e{}s"]
    # ent index in new sentences
    ent_index_pos = []
    new_ner_id2pos = dict()
    for i in range(len(ner_tokens)):
        if ner_pos_list[i][3] != -1:
            continue
        replace_flag = False
        for ent_marker in ent_marker_list:
            cur_marker = ent_marker.format(i)
            if cur_marker in tokens:
                ent_idx = tokens.index(cur_marker)
                ent_index_pos.append((i, ent_idx, cur_marker))
                replace_flag = True
                break
        # marker doesn't exists and ner is none-nested
        if not replace_flag and ner_pos_list[i][3] == -1:
            tokens.append("E{}".format(i))
            ent_index_pos.append((i, len(tokens)-1, "E{}".format(i)))
            print("sentence has error")
            print(tokens)
            error_ids.append(i)
            error_flag = True

    # replace marker from left to right
    ent_index_pos.sort(key=lambda x: x[1])


    for item in ent_index_pos:
        marker_pos = tokens.index(item[2])
        tokens = tokens[:marker_pos] + ner_tokens[item[0]] + tokens[marker_pos + 1:]
        new_ner_id2pos[item[0]] = (marker_pos, marker_pos + len(ner_tokens[item[0]]) - 1)

    return tokens, new_ner_id2pos, error_flag


def translation_scierc(input_path, out_path):
    out_file = open(out_path, "w+", encoding="utf-8")
    sen_cnt = 0
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            print(sen_cnt)
            sen_cnt += 1
            cur_sen = json.loads(line)

            # get the ner pos in sentence
            tokens = cur_sen["tokens"]
            ner = cur_sen["ner"]
            ner_tokens = []
            ner_id2type = dict()
            ner_pos2id = dict()
            ner_id = 0
            for item in ner:
                ner_pos2id[(item[0], item[1])] = ner_id
                ner_tokens.append(tokens[item[0]:item[1] + 1])
                ner_id2type[ner_id] = item[2]
                ner_id += 1
            check_nested(ner)

            rel = cur_sen["relations"]
            id2rel = list()
            for idx in range(len(rel)):
                id2rel.append(
                    [ner_pos2id[(rel[idx][0], rel[idx][1])], ner_pos2id[(rel[idx][2], rel[idx][3])], rel[idx][4]])

            # repalce entity with special marker to
            # maintain entity information during translation.
            replace_ent(tokens, ner)
            sen_str = tokens2str(tokens)
            t1 = translate(sen_str, "en", "zh")
            # translation API limit
            time.sleep(1)
            t2 = translate(t1, "zh", "en")
            translated_tokens = nltk.word_tokenize(t2)

            # replace entity marker with entity tokens
            translated_tokens, new_ner_id2pos = replace_ent_marker(translated_tokens, ner_tokens, ner)
            for i in range(len(ner)):
                if ner[i][3] != -1:
                    h_offset = ner[i][0] - ner[ner[i][3]][0]
                    t_offset = ner[ner[i][3]][1] - ner[i][1]
                    new_ner_id2pos[i] = (new_ner_id2pos[ner[i][3]][0] + h_offset,  new_ner_id2pos[ner[i][3]][1] - t_offset)

            # replace ner and relations with new position
            relations = list()
            for item in id2rel:
                cur_res = []
                cur_res += list(new_ner_id2pos[item[0]])
                cur_res += list(new_ner_id2pos[item[1]])
                cur_res.append(item[2])
                relations.append(cur_res)

            cur_sen["relations"] = relations
            cur_sen["tokens"] = translated_tokens

            time.sleep(1)

            print(json.dumps(cur_sen), file=out_file)


def translation_semeval(input_path, out_path):
    out_file = open(out_path, "w+", encoding="utf-8")
    doc_cnt = 0
    error_ids = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            print(doc_cnt)
            if doc_cnt < 7279:
                doc_cnt += 1
                continue
            cur_data = json.loads(line)
            tokens = cur_data["token"]
            h_start = cur_data["h"]["pos"][0]
            h_end = cur_data["h"]["pos"][1]
            t_start = cur_data["t"]["pos"][0]
            t_end = cur_data["t"]["pos"][1]
            ner_tokens = [cur_data["token"][h_start:h_end+1], cur_data["token"][t_start:t_end+1]]
            ner_pos = [[h_start, h_end, -1, -1],
                       [t_start, t_end, -1, -1]]

            replace_ent(tokens, ner_pos)
            token_str = tokens2str(tokens)

            t1 = translate(token_str, "en", "zh")
            time.sleep(1)
            t2 = translate(t1, "zh", "en")

            translated_tokens = nltk.word_tokenize(t2)
            translated_tokens, new_ner_id2pos, error_flag = replace_ent_marker(translated_tokens, ner_tokens, ner_pos)
            if error_flag:
                error_ids.append(doc_cnt)

            cur_data["token"] = translated_tokens
            cur_data["h"]["pos"] = [new_ner_id2pos[0][0], new_ner_id2pos[0][1]]
            cur_data["t"]["pos"] = [new_ner_id2pos[1][0], new_ner_id2pos[1][1]]
            print(json.dumps(cur_data), file=out_file)
            doc_cnt += 1
            time.sleep(1)
    print(error_ids)
    out_file.close()


if __name__ == "__main__":
    # translation_scierc("../data/scierc_sentence/test_data.json", "../data/scierc_sentence/test_data_res.json")
    # translation_scierc("../data/scierc_sentence/dev.json", "../data/strong_scierc/dev.json")
    # translation_scierc("../data/scierc_sentence/train.json", "../data/strong_scierc/train.json")

    translation_semeval("../data/processed_parse_res.json", "../data/strong_parse_res_append1.json")