from config import VN_TREEBANK_PATH
from common.langs.vi_VN.utils import remove_tone_marks
from os import path, listdir
import random
import string

VN_TREEBANK_FILES = [path.join(VN_TREEBANK_PATH, file) for file in listdir(VN_TREEBANK_PATH)]
# print('%s files' % len(TREEBANK_FILES))

def load_treebank_dataset():
    sents = []
    tags = []
    for item in VN_TREEBANK_FILES:
        item_sents, item_tags = process_treebank_file(item)
        sents.extend(item_sents)
        tags.extend(item_tags)
    return sents, tags

def random_remove_marks(input_str, ratio=0.7):
    result_str = input_str.split()
    for idx, token in enumerate(result_str):
        if random.random() <= ratio:
            result_str[idx] = remove_tone_marks(token)
    return ' '.join(result_str)

def process_treebank_file(filename, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
    print('Reading %s' % filename)
    with open(filename, 'r', encoding='utf-8') as input_file:
        sents = input_file.read().strip().split("\n\n")
    sents = [
        [line.strip().split('\t')[0] for line in sent.strip().split('\n')]
        for sent in sents
    ]
    targets = []
    for sent in sents:
        line_target = []
        for idx, token in enumerate(sent):
            count = token.count(' ')
            if count == 0:
                line_target.append(0)
            else:
                line_target.extend([1] * count + [0])
        targets.append(line_target)
    sents = [' '.join(sent).split(' ') for sent in sents]
    return sents, targets

def reconstruct_sent(sent, target):
    sent_arr = sent
    result_sent = ''
    for idx, token in enumerate(sent_arr):
        if idx > 0 and idx != len(target):
            if target[idx - 1] == 0:
#                 if sent_arr[idx - 1] != "'" and token not in string.punctuation:
                result_sent += ' '
            else:
                result_sent += '_'
        result_sent += token
    return result_sent.strip()