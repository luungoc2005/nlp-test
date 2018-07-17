from common.langs.vi_VN.utils import remove_tone_marks
from os import path, listdir

from os import path, listdir

VN_TREEBANK_PATH = '/Users/2359media/Documents/Samples/treebank/treebank'
VN_TREEBANK_FILES = [path.join(VN_TREEBANK_PATH, file) for file in listdir(VN_TREEBANK_PATH)]
# print('%s files' % len(TREEBANK_FILES))

def random_remove_marks(input_str, ratio=0.7):
    result_str = input_str.split()
    for idx, token in enumerate(result_str):
        if random.random() <= ratio:
            result_str[idx] = remove_tone_marks(token)
    return ' '.join(result_str)

