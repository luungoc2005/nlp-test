
import cython
cimport cython

import numpy as np
from config import NGRAM_BINS
from nltk.tokenize import RegexpTokenizer
from hashlib import md5
from glove_utils import get_text_to_ix

_rt = RegexpTokenizer(r'[a-zA-Z]+|\d+|[^a-zA-Z\d\s]+')
def wordpunct_tokenize(str sent):
    return _rt.tokenize(sent)

def md5_hash_function(str word, bint lower=True):
    cdef str hash_word
    hash_word = word.lower() if lower else word
    return int(md5(hash_word.encode()).hexdigest(), 16)

def bigram_hash(list sequence, int idx, int buckets, bint lower=True):
    cdef str bigram
    if idx - 1 >= 0:
        bigram = sequence[idx - 1] + sequence[idx]
        if lower: bigram = bigram.lower()
        return (hash(bigram) % (buckets - 1) + 1)
    else:
        return 0

def prepare_sequence(list seq, to_ix):
    cdef int unk_token
    unk_token = 0
    idxs = [to_ix.get(w, unk_token) for w in seq]
    return idxs

def sentence_vector(str sentence):
    cdef list token, words_sequence, ngrams_sequence
    tokens = wordpunct_tokenize(sentence)
    words_sequence = prepare_sequence(tokens, get_text_to_ix())
    ngrams_sequence = []

    # n-gram features
    # TODO: experiment with bigrams + trigrams - only bigrams for now
    cdef int idx
    for idx in range(len(tokens)):
        ngrams_sequence.append(bigram_hash(tokens, idx, NGRAM_BINS))
    
    return (words_sequence, ngrams_sequence)

def _process_sentences(list sentences):
    cdef list words_seqs, ngrams_seqs
    cdef int max_words, max_ngrams

    words_seqs = []
    ngrams_seqs = []
    max_words = 0
    max_ngrams = 0

    cdef str sent
    cdef list words_seq, ngrams_seq
    for sent in sentences:
        words_seq, ngrams_seq = sentence_vector(sent)

        if len(words_seq) > max_words:
            max_words = len(words_seq)
        if len(ngrams_seq) > max_ngrams:
            max_ngrams = len(ngrams_seq)
        
        words_seqs.append(words_seq)
        ngrams_seqs.append(ngrams_seq)
    
    cdef list result_words, result_ngrams
    result_words = []
    result_ngrams = []

    cdef list word_seq
    for word_seq in words_seqs:
      new_seq = np.zeros((1, max_words))
      new_seq[:,:len(word_seq)] = word_seq
      result_words.append(new_seq)

    for ngram_seq in ngrams_seqs:
      new_seq = np.zeros((1, max_ngrams))
      new_seq[:,:len(ngram_seq)] = ngram_seq
      result_ngrams.append(new_seq)

    return np.concatenate(np.array(result_words), axis=0), \
           np.concatenate(np.array(result_ngrams), axis=0)