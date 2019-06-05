from pymagnitude import *
from config import MAGNITUDE_PATH, MAX_NUM_WORDS, BASE_PATH, WORDS_SHORTLIST, LANGUAGE
from os import path
import warnings

lazy_loading = -1 if path.isfile(path.join(BASE_PATH, 'DEBUG')) else WORDS_SHORTLIST

vectors = {}

for language_code in MAGNITUDE_PATH.keys():
    if path.exists(MAGNITUDE_PATH[language_code]):
        # print(MAGNITUDE_PATH[language_code])
        # print(MAGNITUDE_PATH[language_code])
        vectors[language_code] = Magnitude(MAGNITUDE_PATH[language_code], lazy_loading=lazy_loading)

# assert len(vectors) > 0, 'Error: No word vector files exist.'

if len(vectors) == 0:
    warnings.warn('Error: No word vector files exist.')
else:
    print('Word vectors data exists for the following languages: %s' % ', '.join(vectors.keys()))
# vectors = Magnitude(MAGNITUDE_PATH, lazy_loading=1, case_insensitive=True)

def get_magnitude_object(language=None):
    return vectors[language if language is not None else LANGUAGE]

def get_emb_matrix(language=None):
    return vectors[language if language is not None else LANGUAGE].get_vectors_mmap()[:MAX_NUM_WORDS]

def get_word_vector(word, *args, language=None, **kwargs):
    return vectors[language if language is not None else LANGUAGE].query(word, *args, **kwargs)

def most_similar(word, topk=5, language=None):
    return vectors[language if language is not None else LANGUAGE].most_similar(word, topn=topk)

def get_dim(language=None):
    return vectors[language if language is not None else LANGUAGE].dim