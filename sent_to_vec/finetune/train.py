
import torch
import torch.nn as nn
import numpy as np
from common.torch_utils import to_gpu
from common.wrappers import ILearner
from common.metrics import accuracy, recall, precision, f1
from config import LM_VOCAB_SIZE, LM_HIDDEN_DIM, LM_SEQ_LEN, LM_EMBEDDING_DIM
from sent_to_vec.masked_lm.data import collate_seq_lm_fn
from typing import Union, Tuple, Iterable

class DoubleClassificationLearner(ILearner):

    def __init__(self, model, *args, **kwargs):
        super(DoubleClassificationLearner, self).__init__(
            model, *args, 
            preprocess_batch=True,
            auto_optimize=True,
            collate_fn=collate_seq_lm_fn,
            **kwargs)