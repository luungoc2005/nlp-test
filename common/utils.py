import torch
import torch.autograd as autograd
import numpy as np
from config import EMBEDDING_DIM # FASTTEXT_BIN
from nltk.tokenize import RegexpTokenizer
# from fastText import FastText
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

"""
Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
"""
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

"""
Find letter index from all_letters, e.g. "a" = 0
"""
def letterToIndex(letter):
    return all_letters.find(letter)

"""
Turns a letter into a <1 x n_letters> Tensor
"""
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

"""
Turn a line into a <line_length x 1 x n_letters>,
or an array of one-hot letter vectors
"""
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

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

"""
returns the topk as a python list
"""
def topk(vec, k):
    vec = torch.topk(vec, k)
    return vec.view(-1).data.tolist()

""" 1-hot encodes a tensor """
def to_categorical(y, num_classes):
    arr = np.eye(num_classes)[y]
    tensor = torch.LongTensor(arr)
    return autograd.Variable(tensor)

def prepare_sequence(seq, to_ix):
    unk_token = 0
    idxs = [to_ix.get(w, unk_token) for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor, requires_grad=False)

def prepare_vec_sequence(seq, to_vec, maxlen=None, output='variable'):
    idxs = np.array([to_vec(w) for w in seq])
    if maxlen:
        seqs = np.zeros((maxlen, idxs.shape[-1]))
        idxs = idxs[-maxlen:]
        seqs[:len(idxs)] = idxs
        idxs = seqs
    if output == 'numpy':
        return idxs
    else:
        tensor = torch.from_numpy(idxs).type(torch.FloatTensor) # Forcefully convert to Float tensor
        if output == 'variable':
            return autograd.Variable(tensor, requires_grad=False)
        elif output == 'tensor':
            return tensor
        else:
            raise NotImplementedError

def to_variable(array, tensor_type=torch.LongTensor):
    return autograd.Variable(tensor_type(array))

"""
Compute log sum exp in a numerically stable way for the forward algorithm
"""
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

from common.glove_utils import get_word_vector
# fastText_model = None

def word_to_vec(word):
    # global fastText_model
    word_vector = get_word_vector(word)
    if word_vector is None:
        return np.zeros(EMBEDDING_DIM) # return all <UNK> as zeros

        # return <UNK> as standard normal
        # if word.strip() == '': # if word is all spaces then return as zeros
        #     return np.zeros(EMBEDDING_DIM)
        # else:
        #     return np.random.randn(EMBEDDING_DIM)

        # if not fastText_model:
        #     print('Loading fastText model for out-of-vocabulary word %s...' % word, end='', flush=True)
        #     fastText_model = FastText.load_model(FASTTEXT_BIN)
        #     print('Done.')
        # word_vector = fastText_model.get_word_vector(word)
    
    return word_vector

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
_rts = RegexpTokenizer(r'[a-zA-Z]+|\d+|\s+|[^a-zA-Z\d\s]+')
def wordpunct_space_tokenize(sent):
    return _rts.tokenize(sent)

_rt = RegexpTokenizer(r'[a-zA-Z]+|\d+|[^a-zA-Z\d\s]+')
def wordpunct_tokenize(sent):
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
