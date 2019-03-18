from pymagnitude import *
from config import MAGNITUDE_PATH, MAX_NUM_WORDS, BASE_PATH, WORDS_SHORTLIST, LANGUAGE
from os import path

lazy_loading = -1 if path.isfile(path.join(BASE_PATH, 'DEBUG')) else WORDS_SHORTLIST

vectors = {
    'en': Magnitude(MAGNITUDE_PATH['en'], lazy_loading=lazy_loading),
    'vi': Magnitude(MAGNITUDE_PATH['vi'], lazy_loading=lazy_loading)
}
# vectors = Magnitude(MAGNITUDE_PATH, lazy_loading=1, case_insensitive=True)

def get_emb_matrix():
    return vectors[LANGUAGE].get_vectors_mmap()[:MAX_NUM_WORDS]

def get_word_vector(word, *args, **kwargs):
    return vectors[LANGUAGE].query(word, *args, **kwargs)

def get_dim():
    return vectors[LANGUAGE].dim