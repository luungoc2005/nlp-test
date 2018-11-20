import torch
import torch.nn as nn
import numpy as np
from common.torch_utils import to_gpu
from common.wrappers import ILearner
from common.metrics import accuracy, recall, precision, f1
from common.utils import to_categorical
from config import LM_VOCAB_SIZE, LM_HIDDEN_DIM, LM_SEQ_LEN, LM_EMBEDDING_DIM
from common.splitcross import SplitCrossEntropyLoss
from sent_to_vec.masked_lm.data import collate_seq_lm_fn
from typing import Union, Tuple, Iterable

class LanguageModelLearner(ILearner):

    def __init__(self, model, *args, **kwargs):
        config = model.config or dict()
        self.seq_len = config.get('seq_len', LM_SEQ_LEN)

        super(LanguageModelLearner, self).__init__(
            model, *args, 
            preprocess_batch=True, 
            auto_optimize=True,
            collate_fn=collate_seq_lm_fn,
            **kwargs)

    def on_training_start(self):
        config = self.model_wrapper.config or dict()
        # hidden_dim = config.get('hidden_dim', LM_HIDDEN_DIM)

        num_words = config.get('num_words', self.model_wrapper.featurizer.tokenizer.num_words)
        use_adasoft = config.get('use_adasoft', True)

        print('Number of tokens', num_words)
        if use_adasoft:
            splits = []
            if num_words > 500000:
                # One Billion
                # This produces fairly even matrix mults for the buckets:
                # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
                splits = [4200, 35000, 180000]
            elif num_words > 75000:
                # WikiText-103
                splits = [2800, 20000, 76000]
            else:
                splits = [num_words // 3, num_words // 3]
        
            print('Cross Entropy Splits: Using', splits)

            self.model_wrapper.config['adasoft_cutoffs'] = splits

            self.criterion = to_gpu(SplitCrossEntropyLoss(hidden_dim, splits))
        else:
            self.criterion = to_gpu(nn.CrossEntropyLoss(ignore_index=0))
        
        self.model_wrapper.config['num_words'] = num_words

        # regularization
        self.clip_grad = config.get('clip_grad', .25)
        self.alpha = config.get('alpha', 2)
        self.beta = config.get('beta', 1)
        self.batch_size = 0
        self.use_adasoft = use_adasoft

    def on_model_init(self):
        print(self.model_wrapper.model)

    def on_epoch(self, X, y):
        batch_size = X.size(1)
        hidden = self.model_wrapper.model.init_hidden(batch_size)

        logits, hidden, rnn_hs, dropped_rnn_hs = self.model_wrapper.model(X, hidden, training=True)

        if self.use_adasoft:
            decoder = self.model_wrapper.model.decoder
            loss = self.criterion(
                decoder.weight,
                decoder.bias,
                logits,
                y
            )
        else:
            loss = self.criterion(
                logits.view(logits.size(0) * logits.size(1), logits.size(2)), 
                y
            )
        
        # Activiation Regularization
        if self.alpha: loss = loss + sum(self.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if self.beta: loss = loss + sum(self.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        
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