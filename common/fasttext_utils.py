"""
Merely a clone of GLOVE_UTILS for FastText instead
This is done for backwards compatibility
"""


import numpy as np
import joblib

from os import path
from config import FASTTEXT_VEC

FASTTEXT_DATA = None

def init_fasttext():
    global FASTTEXT_DATA
    if not FASTTEXT_DATA:
        if path.isfile(FASTTEXT_VEC + '.pickle'):
            file_path = FASTTEXT_VEC + '.pickle'
            print('Importing %s...' % file_path)
            with open(file_path, 'rb') as pickle_file:
                FASTTEXT_DATA = joblib.load(pickle_file)
        else:
            file_path = FASTTEXT_VEC + '.vec'
            print('Importing %s...' % file_path)
            FASTTEXT_DATA = {}
            with open(file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as lines:
                for line in lines:
                    tokens = line.rstrip().split(' ')
                    FASTTEXT_DATA[tokens[0]] = np.array(list(map(float, tokens[1:])))
            with open(FASTTEXT_VEC + '.pickle', 'wb') as pickle_file:
                joblib.dump(FASTTEXT_DATA, pickle_file, compress=3)
    return FASTTEXT_DATA

def get_word_vector(word):
    global FASTTEXT_DATA
    if not FASTTEXT_DATA:
        init_fasttext()
    return FASTTEXT_DATA.get(word, None)

def measure_dist(word1, word2, word_to_vec=None):
    global FASTTEXT_DATA
    if not FASTTEXT_DATA:
        init_fasttext()
    if word_to_vec:
        vec1 = word_to_vec(word1) if type(word1) is str else word1
        vec2 = word_to_vec(word2) if type(word2) is str else word2
    else:
        vec1 = get_word_vector(word1) if type(word1) is str else word1
        vec2 = get_word_vector(word2) if type(word2) is str else word2
    return np.linalg.norm(vec1 - vec2)

def print_top_similar(word, count=15, word_to_vec=None):
    global FASTTEXT_DATA
    if not FASTTEXT_DATA:
        init_fasttext()
    all_words = list([
        [test_word, measure_dist(word, test_word, word_to_vec)]
        for test_word in FASTTEXT_DATA.keys()
    ])
    all_words = sorted(all_words, key=lambda item: item[1])
    return all_words[0:count]
