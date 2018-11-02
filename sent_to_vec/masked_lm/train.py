import torch
import torch.nn as nn
import numpy as np
from common.torch_utils import to_gpu
from common.wrappers import ILearner
from common.metrics import accuracy, recall, precision, f1
from common.utils import to_categorical
from config import LM_VOCAB_SIZE, LM_HIDDEN_DIM, LM_SEQ_LEN
from common.splitcross import SplitCrossEntropyLoss
from typing import Union, Tuple, Iterable

class LanguageModelLearner(ILearner):

    def __init__(self, model, *args, **kwargs):
        config = model.config or dict()
        self.seq_len = config.get('seq_len', LM_SEQ_LEN)

        super(LanguageModelLearner, self).__init__(
            model, *args, 
            preprocess_batch=True, 
            auto_optimize=True,
            **kwargs)

    def on_training_start(self):
        config = self.model_wrapper.config or dict()
        embedding_dim = config.get('embedding_dim', LM_HIDDEN_DIM)

        num_words = config.get('num_words', LM_VOCAB_SIZE)
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

        self.criterion = to_gpu(SplitCrossEntropyLoss(
            embedding_dim,
            splits=splits,
            verbose=False
        ))

        self.hidden = None

        # regularization
        self.clip_grad = config.get('clip_grad', .25)
        self.alpha = config.get('alpha', 2)
        self.beta = config.get('beta', 1)
        self.batch_size = 0

    def on_epoch(self, X, y):
        batch_size = X.size(1)
        hidden = self.model_wrapper.model.init_hidden(batch_size)

        logits, hidden, raw_outputs, outputs = \
            self.model_wrapper.model(X, hidden, return_raws=True)

        loss = self.criterion(
            self.model_wrapper.model.decoder.weight, 
            self.model_wrapper.model.decoder.bias,
            logits,
            y
        )
        
        # Activiation Regularization
        if self.alpha: loss = loss + sum(self.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in outputs[-1:])
        # Temporal Activation Regularization (slowness)
        if self.beta: loss = loss + sum(self.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in raw_outputs[-1:])
        
        loss.backward()

        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model_wrapper.model.parameters(), 
                self.clip_grad
            )
        
        return {
            'loss': loss.detach().cpu().item()
        }

    def calculate_metrics(self, logits, y):
        return None