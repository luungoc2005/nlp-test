import numpy as np
import joblib

from os import path
from config import GLOVE_PATH

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
                GLOVE_DATA = {
                    line.split()[0]: np.array(list(map(float, line.split()[1:])), dtype='float32')
                    for line in lines
                }
            with open(GLOVE_PATH + '.pickle', 'wb') as pickle_file:
                joblib.dump(GLOVE_DATA, pickle_file, compress=3)
    return GLOVE_DATA

def measure_dist(word1, word2):
    global GLOVE_DATA
    if not GLOVE_DATA:
        init_glove()
    return np.linalg.norm(GLOVE_DATA[word1] - GLOVE_DATA[word2])

def print_top_similar(word, count=15):
    global GLOVE_DATA
    if not GLOVE_DATA:
        init_glove()
    all_words = list([
        [test_word, np.linalg.norm(GLOVE_DATA[test_word] - GLOVE_DATA[word])]
        for test_word in GLOVE_DATA.keys()
    ])
    all_words = sorted(all_words, key=lambda item: item[1])
    return all_words[0:count]
