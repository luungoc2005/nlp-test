
import cython
cimport cython

import numpy as np
import joblib

import io

from os import path
from config import SKIP_FIRST_LINE, GLOVE_PATH, MAX_NUM_WORDS, CACHE_DATA, EMBEDDING_DIM

GLOVE_DATA = None
WORDS_DICT = None
EMB_MATRIX = None

def init_glove():
    global GLOVE_DATA, GLOVE_PATH
    cdef str file_path, line, pickle_path
    cdef int line_count

    pickle_path = GLOVE_PATH + '.pickle'
    if GLOVE_DATA is None:
        if path.isfile(pickle_path):
            print('Importing %s...' % pickle_path)
            with open(pickle_path, 'rb') as pickle_file:
                GLOVE_DATA = joblib.load(pickle_file)
        else:
            print('Importing %s...' % GLOVE_PATH)
            fin = io.open(GLOVE_PATH, 'r', encoding='utf-8', newline='\n', errors='ignore')
            
            if SKIP_FIRST_LINE:
                n, d = map(int, fin.readline().split())
            
            line_count = 0
            GLOVE_DATA = {}

            for line in fin:
                line_arr = line.rstrip().split(' ')
                GLOVE_DATA[line_arr[0]] = np.array(list(map(float, line_arr[1:])))
                line_count += 1
                if line_count >= MAX_NUM_WORDS: break
            
            if CACHE_DATA:
                with open(pickle_path, 'wb') as pickle_file:
                    joblib.dump(GLOVE_DATA, pickle_file, compress=3)
    return GLOVE_DATA

def get_emb_matrix():
    global GLOVE_DATA, EMB_MATRIX
    cdef int idx
    if EMB_MATRIX is not None:
        return EMB_MATRIX
    else:
        if GLOVE_DATA is None:
            init_glove()
        # idx 0 will be the <UNK> token
        EMB_MATRIX = np.zeros((MAX_NUM_WORDS + 1, EMBEDDING_DIM))

        for idx, val in enumerate(GLOVE_DATA.values()):
            EMB_MATRIX[idx + 1] = val
        return EMB_MATRIX

def get_text_to_ix():
    global WORDS_DICT, GLOVE_DATA
    cdef int count
    cdef str word

    if WORDS_DICT is not None:
        return WORDS_DICT
    else:
        count = 1 # starts from 1
        if GLOVE_DATA is None:
            init_glove()

        WORDS_DICT = {}
        for word in GLOVE_DATA.keys():
            WORDS_DICT[word] = count
            count += 1
    
    return WORDS_DICT

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

def get_glove_data():
    global GLOVE_DATA
    if not GLOVE_DATA:
        init_glove()
    return GLOVE_DATA