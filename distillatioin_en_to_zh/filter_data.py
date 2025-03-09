import json
import re
import os
from tqdm import tqdm

def contains_chinese(text):
    # 使用正则表达式检查是否存在中文字符
    return re.search(r'[\u4e00-\u9fff]', text) is not None

def load_data(infile):
    valid_data = []
    unvalid_data = []
    with open(infile, "r") as f:
        for line in tqdm(f.readlines(), desc=f"Loading data {os.path.basename(infile)}"):
            data = json.loads(line)
            if contains_chinese(data['english']):
                unvalid_data.append(data)
                continue
            valid_data.append(data)
    print(f"Loaded {len(valid_data)} data from {os.path.basename(infile)}")
    return valid_data, unvalid_data

def main():
    train_data_file = 'dataset/translation2019zh/translation2019zh_train.json'
    test_data_file = 'dataset/translation2019zh/translation2019zh_valid.json'
    train_data, unvalid_train_data = load_data(train_data_file)
    test_data, unvalid_test_data = load_data(test_data_file)
    with open('dataset/translation2019zh/translation2019zh_train_clean.jsonl', 'w') as f:
        for data in train_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    with open('dataset/translation2019zh/translation2019zh_valid_clean.jsonl', 'w') as f:
        for data in test_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    with open('dataset/translation2019zh/translation2019zh_train_unvalid.jsonl', 'w') as f:
        for data in unvalid_train_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    with open('dataset/translation2019zh/translation2019zh_valid_unvalid.jsonl', 'w') as f:
        for data in unvalid_test_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    return
