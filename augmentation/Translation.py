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
    return requests.get(translation_url, params=para)


if __name__ == "__main__":
    # print(json.dumps(translate("hello world", "en", "zh")))

    print(translate("hello world", "en", "zh").json())


