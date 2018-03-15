import torch
import torch.autograd as autograd
import numpy as np
from config import FASTTEXT_PATH
from nltk.tokenize import RegexpTokenizer
from fastText import FastText

"""
returns a python float
"""
def to_scalar(var):
    return var.view(-1).data.tolist()[0]

"""
returns the argmax as a python int
"""
def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

def process_input(data):
    return [
        (wordpunct_space_tokenize(sent), tags.split())
        for (sent, tags) in data
    ]

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

def prepare_vec_sequence(seq, to_vec):
    idxs = np.array([to_vec(w) for w in seq])
    tensor = torch.from_numpy(idxs)
    return autograd.Variable(tensor)

"""
Compute log sum exp in a numerically stable way for the forward algorithm
"""
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


WORD_EMBEDDINGS = {}
fastText_model = None

def word_to_vec(word):
    global fastText_model, WORD_EMBEDDINGS
    if not fastText_model:
        print('Loading fastText model...', end='', flush=True)
        fastText_model = FastText.load_model(FASTTEXT_PATH)
        print('Done.')
    
    if word not in WORD_EMBEDDINGS:
        WORD_EMBEDDINGS[word] = fastText_model.get_word_vector(word)
    return WORD_EMBEDDINGS[word]

def get_datetime_hostname():
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    return current_time + '_' + socket.gethostname()

"""
Small modification to NLTK wordpunct_tokenize
Because NLTK's won't split on '_' or numbers

Reference: https://github.com/nltk/nltk/issues/1900

This function will treat spaces as separate tokens as well
Which helps with reconstruction.

Use nltk function instead for testing with tagging datasets
"""
_rt = RegexpTokenizer(r'[a-zA-Z]+|\d+|\s+|[^a-zA-Z\d\s]+')
def wordpunct_space_tokenize(sent):
    return _rt.tokenize(sent)

import time
import math

# Helper functions for time remaining
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    if percent != 0:
        es = s / (percent)
        rs = es - s
    else:
        rs = 0
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))