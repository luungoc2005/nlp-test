import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.sparse
import numpy as np
from hashlib import md5
from common.utils import wordpunct_tokenize, prepare_vec_sequence, word_to_vec
from config import NGRAM_BINS, EMBEDDING_DIM, SENTENCE_DIM

# Just in case
def md5_hash_function(word, lower=True):
    hash_word = word.lower() if lower else word
    return int(md5(hash_word.encode()).hexdigest(), 16)

def bigram_hash(sequence, idx, buckets, lower=True):
    if idx - 1 >= 0:
        bigram = sequence[idx - 1] + sequence[idx]
        if lower: bigram = bigram.lower()
        return (hash(bigram) % (buckets - 1) + 1)
    else:
        return 0

def sentence_vector(sentence, maxlen=None):
    result = np.zeros(EMBEDDING_DIM + NGRAM_BINS)
    tokens = wordpunct_tokenize(sentence)
    vec_sequence = prepare_vec_sequence(tokens, word_to_vec, SENTENCE_DIM, output='numpy')
    result[:EMBEDDING_DIM] = np.mean(vec_sequence, axis=0)

    # n-gram features
    # TODO: experiment with bigrams + trigrams - only bigrams for now
    # idxs = []
    for idx in range(len(tokens)):
        result[EMBEDDING_DIM + bigram_hash(tokens, idx, NGRAM_BINS)] = 1.
        # idxs.append(bigram_hash(tokens, idx, NGRAM_BINS))
    # values = np.ones(len(idxs))
    return result
    # return torch.from_numpy(result).float()

class FastText(nn.Module):

    def __init__(self,
                 input_size=None,
                 hidden_size=100,
                 classes=10):
        super(FastText, self).__init__()

        self.input_size = input_size or EMBEDDING_DIM + NGRAM_BINS
        self.hidden_size = hidden_size
        self.classes = classes

        self.i2h = nn.Linear(self.input_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, self.classes)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal(self.h2o.weight)
        self.h2o.bias.data.fill_(0)

    def forward(self, sequence):
        x = self.i2h(sequence)
        x = self.h2o(x)
        return x