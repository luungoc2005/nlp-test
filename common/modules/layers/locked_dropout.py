import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Union
from config import EMBEDDING_DIM, UNK_TAG
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from common.torch_utils import to_gpu
from common.utils import letterToIndex, n_letters, prepare_vec_sequence, word_to_vec, argmax

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        m.requires_grad = False
        mask = m / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x
