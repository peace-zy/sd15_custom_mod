#-*-coding: utf-8-*-
import json
import copy
from cv2 import add
from tqdm import tqdm
import re
from transformers import BertTokenizer, BertModel, CLIPTokenizer, CLIPTextModel

def read_teacher_file(teacher_jsonl_file):
    with open(teacher_jsonl_file, "r") as f:
        for line in f:
            data = eval(line)
            yield data

def read_student_file(student_jsonl_file):
    with open(student_jsonl_file, "r") as f:
        for line in f:
            data = eval(line)
            yield data

def is_pure_chinese_char(char):
    """
    常用汉字：U+4E00 到 U+9FFF
    扩展A区汉字：U+3400 到 U+4DBF
    扩展B区汉字：U+20000 到 U+2A6DF
    扩展C区汉字：U+2A700 到 U+2B73F
    扩展D区汉字：U+2B740 到 U+2B81F
    扩展E区汉字：U+2B820 到 U+2CEAF
    扩展F区汉字：U+2CEB0 到 U+2EBEF
    扩展G区汉字：U+30000 到 U+3134F
    """
    if '\u4e00' <= char <= '\u9fff':
        return True
    if '\u3400' <= char <= '\u4dbf':
        return True
    if '\u20000' <= char <= '\u2a6df':
        return True
    if '\u2a700' <= char <= '\u2b73f':
        return True
    if '\u2b740' <= char <= '\u2b81f':
        return True
    if '\u2b820' <= char <= '\u2ceaf':
        return True
    if '\u2ceb0' <= char <= '\u2ebef':
        return True
    if '\u30000' <= char <= '\u3134f':
        return True
    return False

def is_chinese_char(char):
    """检查是否是中文字符或标点符号"""
    if '\u4e00' <= char <= '\u9fff':
        return True
    return False
    if '\u3000' <= char <= '\u303f':
        return True
    if '\u3400' <= char <= '\u4dbf':
        return True
    if '\u20000' <= char <= '\u2a6df':
        return True
    if '\u2a700' <= char <= '\u2b73f':
        return True
    if '\u2b740' <= char <= '\u2b81f':
        return True
    if '\u2b820' <= char <= '\u2ceaf':
        return True
    if '\u2ceb0' <= char <= '\u2ebef':
        return True
    if '\uf900' <= char <= '\ufaff':
        return True
    if '\u2f800' <= char <= '\u2fa1f':
        return True
    return False

def is_numeric_string(s):
    pattern = r'^\d+$'
    res = bool(re.match(pattern, s))
    if res:
        for c in s:
            if c not in "0123456789":
                res = False
                break
    return res

def merge(clip_tokenizer, bert_tokenizer):
    # 获取词汇表
    bert_vocab = bert_tokenizer.get_vocab()
    clip_vocab = clip_tokenizer.get_vocab()

    match_tokens = []
    mismatch_tokens = []
    bert_tokens = [token.replace("##", "") for token in bert_vocab.keys()]
    clip_tokens = [token.replace("</w>", "") for token in clip_vocab.keys()]
    for token in bert_tokens:
        if token in clip_tokens:
            match_tokens.append(token)
        else:
            mismatch_tokens.append(token)
    print(f"Match tokens: {len(match_tokens)}, Mismatch tokens: {len(mismatch_tokens)}, "
          f"Bert tokens: {len(bert_tokens)}, Clip tokens: {len(clip_tokens)}, "
          f"ratio: {len(match_tokens) / len(clip_tokens)}")

    ori_clip_tokenizer = copy.deepcopy(clip_tokenizer)

    chinese_tokens = [token for token in bert_vocab.keys() if any(is_chinese_char(char) for char in token)]
    with open("chinese_tokens.txt", "w") as f:
        for token in chinese_tokens:
            f.write(f"{token}\n")
    print(f"Chinese tokens: {len(chinese_tokens)}")
    unique_chinese_tokens = []
    for token in chinese_tokens:
        token = token.replace("##", "")
        unique_chinese_tokens.append(token)

    print(f"Chinese tokens: {len(unique_chinese_tokens)}")
    digit_tokens = [token for token in bert_vocab.keys() if is_numeric_string(token) and len(token) > 1]
    with open("digit_tokens.txt", "w") as f:
        for token in digit_tokens:
            f.write(f"{token}\n")

    additional_tokens = unique_chinese_tokens + digit_tokens
    additional_tokens = unique_chinese_tokens
    # 添加中文单字和数字到词汇表中
    for idx, token in enumerate(additional_tokens):
        if token not in clip_vocab and f"{token}</w>" not in clip_vocab:
            clip_vocab[token] = len(clip_vocab)

    # 更新tokenizer的词汇表
    clip_tokenizer.add_tokens(additional_tokens)


    save_path = "./merged_tokenizer_new_1"
    # 保存修改后的词汇表
    clip_tokenizer.save_pretrained(save_path)

    # 重新加载修改后的tokenizer
    merged_tokenizer = CLIPTokenizer.from_pretrained(save_path)
    print(f"vocab size: {merged_tokenizer.vocab_size}")

    # 测试新的tokenizer
    text = "This is a test sentence."
    text = "你好，我的小名叫小明，年龄169岁, 10⁶km"
    #text = "中国的中".encode("utf-8").decode("utf-8")
    tokens = merged_tokenizer.tokenize(text)
    token_ids = merged_tokenizer.convert_tokens_to_ids(tokens)

    print("Tokens:", tokens)
    print("Token IDs:", token_ids)
    print("Token Size:", len(token_ids))
    ori_tokens = bert_tokenizer.tokenize(text)
    ori_token_ids = bert_tokenizer.convert_tokens_to_ids(ori_tokens)
    print("Original Tokens:", ori_tokens)
    print("Original Token IDs:", ori_token_ids)
    print("Original Token Size:", len(ori_token_ids))

    clip_tokens = ori_clip_tokenizer.tokenize(text)
    clip_token_ids = ori_clip_tokenizer.convert_tokens_to_ids(clip_tokens)
    print("Clip Tokens:", clip_tokens)
    print("Clip Token IDs:", clip_token_ids)
    print("Clip Token Size:", len(clip_token_ids))
    """
    for token in unique_chinese_tokens:
        clip_tokens = ori_clip_tokenizer.tokenize(token)
        if len(clip_tokens) == len(token):
            print(f"[same] token: {token}, clip_tokens: {clip_tokens}")
        else:
            print(f"[diff] token: {token}, clip_tokens: {clip_tokens}")
    """

def main():
    teacher_jsonl_file = "teacher.data"
    student_jsonl_file = "student.data"
    model_path = 'test/realisticVisionV51_v51VAE'
    teacher_tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    student_tokenizer = BertTokenizer.from_pretrained(model_path, subfolder="tokenizer_bert")
    merge(teacher_tokenizer, student_tokenizer)
    return
    teacher_iter = read_teacher_file(teacher_jsonl_file)
    student_iter = read_teacher_file(student_jsonl_file)
    while True:
        try:
            teacher_token_ids = next(teacher_iter)[0]
            teacher_tokens = teacher_tokenizer.convert_ids_to_tokens(teacher_token_ids)
            student_token_ids = next(student_iter)[0]
            student_tokens = student_tokenizer.convert_ids_to_tokens(student_token_ids)
            text = student_tokenizer.convert_tokens_to_string(student_tokens)
            print(f"text={text}\n"
                f"teacher_tokens: {teacher_tokens}\nstudent_tokens: {student_tokens}\n"
                f"teacher_token_ids: {teacher_token_ids}\nstudent_token_ids: {student_token_ids}\n")
        except StopIteration:
            print("Generator has been exhausted.")
            break

    return

if __name__ == "__main__":
    main()
