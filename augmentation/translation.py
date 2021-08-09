import requests
import hashlib
import json
from settings import translation_id, translation_key, salt, translation_url


def translate(query, source_lan, target_lan):
    hash_source = translation_id + query + salt + translation_key
    sign = hashlib.md5(hash_source.encode("utf8")).hexdigest()
    para = dict()
    para["q"] = query
    para["from"] = source_lan
    para["to"] = target_lan
    para["appid"] = translation_id
    para["salt"] = salt
    para["sign"] = sign
    res = requests.get(translation_url, params=para).json()
    # print(res)
    return res["trans_result"][0]["dst"]


if __name__ == "__main__":
    # print(json.dumps(translate("hello world", "en", "zh")))
    a = "在麻省理工学院林肯实验室，我们一直在开发E0 $ E1。"
    # print(translate("E1 ( E2 ) is a key problem in E3 ( E4 ) .", "en", "zh"))
    print(translate(a, "zh", "en"))



