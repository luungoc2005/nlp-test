
import cython
cimport cython

import numpy as np
from config import NGRAM_BINS
from nltk.tokenize import RegexpTokenizer
from hashlib import md5
# from glove_utils import get_text_to_ix
from common.word_vectors import get_word_vector, get_dim

_rt = RegexpTokenizer(r'[a-zA-Z]+|\d+|[^a-zA-Z\d\s]+')
def wordpunct_tokenize(str sent):
    return _rt.tokenize(sent)

def md5_hash_function(str word):
    return int(md5(word.encode()).hexdigest(), 16)

# buckets must be power of 2 - sized
def unigram_hash(list sequence, int idx, int buckets, bint lower=True):
    cdef str unigram
    if lower: unigram = sequence[idx].lower()
    # return (md5_hash_function(unigram) % (buckets - 1) + 1)
    # return (md5_hash_function(unigram) & (buckets - 1))
    return (hash(unigram) & (buckets - 1))

def bigram_hash(list sequence, int idx, int buckets, bint lower=True):
    cdef str bigram
    if idx >= 1:
        bigram = sequence[idx - 1] + ' ' + sequence[idx]
        if lower: bigram = bigram.lower()
        # return (md5_hash_function(bigram) % (buckets - 1) + 1)
        # return (md5_hash_function(bigram) & (buckets - 1))
        return (hash(bigram) & (buckets - 1))
    else:
        return 0

def trigram_hash(list sequence, int idx, int buckets, bint lower=True):
    cdef str trigram
    if idx >= 2:
        trigram = sequence[idx - 2] + ' ' + sequence[idx - 1] + ' ' + sequence[idx]
        if lower: trigram = trigram.lower()
        # return (md5_hash_function(trigram) % (buckets - 1) + 1)
        return (hash(trigram) % (buckets - 1) + 1)
    else:
        return 0

def _process_sentences(list sentences):
    cdef list tokens_list, ngram_idxs
    cdef int max_len, idx
    max_len = 0
    tokens_list = [wordpunct_tokenize(sent) for sent in sentences]
    embeddings = get_word_vector(tokens_list)

    ngram_idxs = []
    cdef list tokens, bigrams
    for tokens in tokens_list:
        bigrams = [bigram_hash(tokens, idx, NGRAM_BINS) for idx, _ in enumerate(tokens)]
        if len(bigrams) > max_len: max_len = len(bigrams)
        ngram_idxs.append(bigrams)

    ngram_ndarray = np.zeros((len(tokens_list), max_len))
    cdef list ngrams_seq
    for idx, ngrams_seq in enumerate(ngram_idxs):
        ngram_ndarray[idx,:len(ngrams_seq)] = ngrams_seq

    return embeddings, ngram_ndarray


