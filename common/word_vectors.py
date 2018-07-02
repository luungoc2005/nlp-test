from pymagnitude import *
from config import MAGNITUDE_PATH, MAX_NUM_WORDS

vectors = Magnitude(MAGNITUDE_PATH)

def get_emb_matrix():
    return vectors.get_vectors_mmap()[:MAX_NUM_WORDS]

def get_word_vector(word):
    return vectors.query(word)
