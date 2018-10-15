import torch
import numpy as np
import torch.nn as nn
from common.torch_utils import to_gpu
from common.wrappers import ILearner
from common.metrics import accuracy, recall, precision, f1
from common.utils import to_categorical
from config import LM_VOCAB_SIZE, LM_HIDDEN_DIM
from typing import Union, Tuple

class LanguageModelLearner(ILearner):

    def __init__(self, model, *args, **kwargs):
        super(LanguageModelLearner, self).__init__(
            model, *args, auto_optimize=True, **kwargs)

    def init_on_data(self, X, y):
        config = self.model_wrapper.config or dict()
        num_words = config.get('num_words', LM_VOCAB_SIZE)
        hidden_size = config.get('hidden_size', LM_HIDDEN_DIM)
        splits = []
        if num_words > 500000:
            # One Billion
            # This produces fairly even matrix mults for the buckets:
            # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
            splits = [4200, 35000, 180000]
        elif num_words > 75000:
            # WikiText-103
            splits = [2800, 20000, 76000]
        print('Cross Entropy Splits: Using', splits)

        self.criterion = nn.AdaptiveLogSoftmaxWithLoss(
            hidden_size, 
            num_words,
            cutoffs=splits
        )
        self.hidden = None
        self.clip_grad = config.get('clip_grad', 5)

    def repackage_hidden(self, h) -> Union[torch.Tensor, Tuple]:
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def on_epoch(self, X, y):
        if self.hidden is None:
            batch_size = X.size(0)
            self.hidden = self.model_wrapper.model.init_hidden(batch_size)
        else:
            self.hidden = self.repackage_hidden(self.hidden)
        
        logits, self.hidden = self.model_wrapper.model(X)
        loss = self.criterion(logits, y)

        loss.backward()

        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm(
                self.model_wrapper.model.parameters(), 
                self.clip_grad
            )
        
        return {
            'loss': loss.detach().item(), 
            'logits': logits.detach()
        }

    def calculate_metrics(self, logits, y):
        return None