from pymagnitude import *
from config import MAGNITUDE_PATH, MAX_NUM_WORDS, BASE_PATH, WORDS_SHORTLIST, LANGUAGE
from os import path

lazy_loading = -1 if path.isfile(path.join(BASE_PATH, 'DEBUG')) else WORDS_SHORTLIST

vectors = {}

for language_code in MAGNITUDE_PATH.keys():
    if path.exists(MAGNITUDE_PATH[language_code]):
        vectors[language_code] = Magnitude(MAGNITUDE_PATH[language_code], lazy_loading=lazy_loading)

assert len(vectors) > 0, 'Error: No word vector files exist.'
print('Word vectors data exists for the following languages: %s' % ', '.join(vectors.keys()))
# vectors = Magnitude(MAGNITUDE_PATH, lazy_loading=1, case_insensitive=True)

def get_emb_matrix():
    return vectors[LANGUAGE].get_vectors_mmap()[:MAX_NUM_WORDS]

def get_word_vector(word, *args, **kwargs):
    return vectors[LANGUAGE].query(word, *args, **kwargs)

def get_dim():
    return vectors[LANGUAGE].dim