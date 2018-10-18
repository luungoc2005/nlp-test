import torch
import torch.autograd as autograd
import numpy as np
from config import EMBEDDING_DIM  # FASTTEXT_BIN
from nltk.tokenize import RegexpTokenizer
# from glove_utils import get_word_vector
from common.word_vectors import get_word_vector
import unicodedata
import string
import time
import math

# from fastText import FastText

# all_letters = string.ascii_letters + " .,;'"
all_letters = string.printable
n_letters = len(all_letters)


def unicodeToAscii(s):
    """
    Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def letterToIndex(letter):
    """
    Find letter index from all_letters, e.g. "a" = 1
    0 will be the OOV index
    """
    return all_letters.find(letter) + 1


def letterToTensor(letter):
    """
    Turns a letter into a <1 x n_letters> Tensor
    """
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line):
    """
    Turn a line into a <line_length x 1 x n_letters>,
    or an array of one-hot letter vectors
    """
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def to_scalar(var):
    """
    returns a python float
    """
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    """
    returns the argmax as a python int
    """
    _, idx = torch.max(vec, -1)
    return to_scalar(idx)


def topk(vec, k):
    """
    returns the topk as a python list
    """
    vec = torch.topk(vec, k)
    return vec.view(-1).data.tolist()


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    arr = np.eye(num_classes)[y]
    tensor = torch.LongTensor(arr)
    return autograd.Variable(tensor)


def prepare_sequence(seq, to_ix):
    unk_token = 0
    idxs = [to_ix.get(w, unk_token) for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor, requires_grad=False)


def prepare_vec_sequence(seq, to_vec, maxlen=None, output='tensor'):
    idxs = np.array([to_vec(w) for w in seq])
    if maxlen:
        seqs = np.zeros((maxlen, idxs.shape[-1]))
        idxs = idxs[-maxlen:]
        seqs[:len(idxs)] = idxs
        idxs = seqs
    if output == 'numpy':
        return idxs
    else:
        tensor = torch.from_numpy(idxs).float()  # Forcefully convert to Float tensor
        tensor.requires_grad = False
        return tensor


def to_variable(array, tensor_type=torch.LongTensor):
    return autograd.Variable(tensor_type(array))


# def log_sum_exp(vec):
#     """
#     Compute log sum exp in a numerically stable way for the forward algorithm
#     """
#     max_score = vec[0, argmax(vec)]
#     max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
#     return max_score + \
#            torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# fastText_model = None


def word_to_vec(word, *args, **kwargs):
    if len(word) == 0: return None

    word_vector = get_word_vector(word, *args, **kwargs)
    if word_vector is None:
        return np.zeros((len(word), EMBEDDING_DIM)) # return all <UNK> as zeros

    return word_vector


def get_datetime_hostname():
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    return current_time + '_' + socket.gethostname()


_rts = RegexpTokenizer(r'[a-zA-Z]+|\d+|\s+|[^a-zA-Z\d\s]+')


def wordpunct_space_tokenize(sent):
    """
    Small modification to NLTK wordpunct_tokenize
    Because NLTK's won't split on '_' or numbers

    Reference: https://github.com/nltk/nltk/issues/1900

    This function will treat spaces as separate tokens as well
    Which helps with reconstruction.

    Use nltk function instead for testing with tagging datasets
    """
    return _rts.tokenize(sent)


_rt = RegexpTokenizer(r'[a-zA-Z]+|\d+|[^a-zA-Z\d\s]+')


def wordpunct_tokenize(sent):
    return _rt.tokenize(sent)


# Helper functions for time remaining
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    if percent != 0:
        es = s / percent
        rs = es - s
    else:
        rs = 0
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/sequence.py
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.
    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.
    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.
    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.
    Pre-padding is the default.
    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.
    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`
    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, str) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x