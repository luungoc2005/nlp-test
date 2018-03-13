import torch
import torch.autograd as autograd
import numpy as np

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