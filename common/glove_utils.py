import numpy as np
import joblib

from os import path
from config import GLOVE_PATH, MAX_NUM_WORDS

GLOVE_DATA = None

def init_glove():
    global GLOVE_DATA, GLOVE_PATH
    if not GLOVE_DATA:
        if path.isfile(GLOVE_PATH + '.pickle'):
            file_path = GLOVE_PATH + '.pickle'
            print('Importing %s...' % file_path)
            with open(file_path, 'rb') as pickle_file:
                GLOVE_DATA = joblib.load(pickle_file)
        else:
            file_path = GLOVE_PATH + '.txt'
            print('Importing %s...' % file_path)
            with open(file_path, 'r') as lines:
                line_count = 0
                GLOVE_DATA = {}
                for line in lines:
                    line_arr = line.split()
                    GLOVE_DATA[line_arr[0]] = np.array(list(map(float, line_arr[1:])))
                    line_count += 1
                    if line_count >= MAX_NUM_WORDS: break
            with open(GLOVE_PATH + '.pickle', 'wb') as pickle_file:
                joblib.dump(GLOVE_DATA, pickle_file, compress=3)
    return GLOVE_DATA

def get_word_vector(word):
    global GLOVE_DATA
    if not GLOVE_DATA:
        init_glove()
    # because GLoVe vectors are uncased
    return GLOVE_DATA.get(word.lower(), None)

def measure_dist(word1, word2):
    vec1 = get_word_vector(word1) if type(word1) is str else word1
    vec2 = get_word_vector(word2) if type(word2) is str else word2
    return np.linalg.norm(vec1 - vec2)

def print_top_similar(word, count=15):
    global GLOVE_DATA
    if not GLOVE_DATA:
        init_glove()
    all_words = list([
        [test_word, measure_dist(word, test_word)]
        for test_word in GLOVE_DATA.keys()
    ])
    all_words = sorted(all_words, key=lambda item: item[1])
    return all_words[0:count]
