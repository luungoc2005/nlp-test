import unicodedata
import random

normalize_map = [
    ("òa", "oà"),
    ("Òa", "Oà"),
    ("óa", "oá"),
    ("ỏa", "oả"),
    ("õa", "oã"),
    ("ọa", "oạ"),
    ("òe", "oè"),
    ("óe", "oé"),
    ("ỏe", "oẻ"),
    ("õe", "oẽ"),
    ("ọe", "oẹ"),
    ("ùy", "uỳ"),
    ("úy", "uý"),
    ("ủy", "uỷ"),
    ("ũy", "uỹ"),
    ("ụy", "uỵ"),
    ("Ủy", "Uỷ")
]

def tone_marks_normalize(input_str):
    result_str = input_str
    for item in normalize_map:
        result_str = result_str.replace(item[0], item[1])
    return result_str

def remove_tone_marks(input_str):
    result_str = ''.join(
        c for c in unicodedata.normalize('NFD', str(input_str).replace('đ', 'd').replace('Đ', 'D'))
        if unicodedata.category(c) != 'Mn'
    )
    return result_str

def random_remove_marks(input_str, ratio=0.3):
    result_str = input_str.split()
    for idx, token in enumerate(result_str):
        if random.random() <= ratio:
            result_str[idx] = remove_tone_marks(token)
    return ' '.join(result_str)