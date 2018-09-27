from pymagnitude import *
from config import MAGNITUDE_PATH, MAX_NUM_WORDS, BASE_PATH, WORDS_SHORTLIST
from os import path

lazy_loading = -1 if path.isfile(path.join(BASE_PATH, 'DEBUG')) else WORDS_SHORTLIST

vectors = {
    'en': Magnitude(MAGNITUDE_PATH['en'], lazy_loading=lazy_loading, case_insensitive=True)
}
# vectors = Magnitude(MAGNITUDE_PATH, lazy_loading=1, case_insensitive=True)

def get_emb_matrix(lang='en'):
    return vectors[lang].get_vectors_mmap()[:MAX_NUM_WORDS]

def get_word_vector(word, lang='en', *args, **kwargs):
    return vectors[lang].query(word, *args, **kwargs)

def get_dim(lang='en'):
    return vectors[lang].dim