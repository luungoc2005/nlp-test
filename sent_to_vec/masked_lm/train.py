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
from sent_to_vec.masked_lm.bert_model import BertLMWrapper
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

        if isinstance(model, BertLMWrapper):
            self.bert_mode = True
            print('Training in BERT mode')

        # regularization
        self.clip_grad = config.get('clip_grad', 5.)
        self.alpha = config.get('alpha', 0)
        self.beta = config.get('beta', 0)
        self.use_adasoft = config.get('use_adasoft', True)

    def on_training_start(self):
        config = self.model_wrapper.config or dict()

        num_words = config.get('num_words', self.model_wrapper.featurizer.tokenizer.num_words)

        print('Number of tokens', num_words)
        
        self.criterion = to_gpu(nn.CrossEntropyLoss(ignore_index=0)) if not self.use_adasoft else None
        
        self.model_wrapper.config['num_words'] = num_words

        self.batch_size = 0

    def on_model_init(self):
        print(self.model_wrapper.model)

    def on_epoch(self, X, y, gradient_accumulation_steps:int = 1.):
        bert_mode = hasattr(self, 'bert_mode') and self.bert_mode
        # TODO: implement masking for BERT
        #     X, bert_mask = X
            
        batch_size = X.size(1)

        model = self.model_wrapper.model

        if hasattr(model, 'init_hidden'):
            hidden = model.init_hidden(batch_size)
            logits, hidden, rnn_hs, dropped_rnn_hs = model(X, hidden, training=True)

        else:
            hidden = None
            logits, hidden, _, _ = model(X, training=True)

        if self.bert_mode:
            decoder = model.cls.predictions.decoder
        else:
            decoder = model.decoder
        if self.use_adasoft:
            loss = model.adasoft(
                decoder.weight,
                decoder.bias,
                logits.view(logits.size(0) * logits.size(1), logits.size(2)),
                y.view(-1)
            )
        else:
            decoded = decoder(logits.view(logits.size(0) * logits.size(1), logits.size(2)))
            loss = self.criterion(
                decoded, 
                y.view(-1)
            )
        
        # Activiation Regularization
        if self.alpha: loss = loss + sum(self.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if self.beta: loss = loss + sum(self.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                self.clip_grad
            )
        
        return {
            'loss': loss
        }

    def calculate_metrics(self, logits, y):
        return None