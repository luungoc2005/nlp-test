import torch
import torch.nn as nn
import numpy as np
from common.torch_utils import to_gpu
from common.wrappers import ILearner
from common.metrics import accuracy, recall, precision, f1
from common.utils import to_categorical
from config import LM_VOCAB_SIZE, LM_HIDDEN_DIM, LM_SEQ_LEN
from torch.utils.data import Dataset
from sent_to_vec.awd_lm.data import read_wikitext
from sent_to_vec.awd_lm.splitcross import SplitCrossEntropyLoss
from typing import Union, Tuple, Iterable

class WikiTextDataset(Dataset):

    def __init__(self):
        super(WikiTextDataset, self).__init__()

    def initialize(self, model_wrapper, data_path, batch_size=64):
        if isinstance(data_path, str):
            self.raw_sents = read_wikitext(data_path)
            print('Loaded {} sentences from {}'.format(len(self.raw_sents), data_path))
        else:
            self.raw_sents = []
            for file_path in data_path:
                file_sents = read_wikitext(file_path)
                self.raw_sents.extend(file_sents)
                print('Loaded {} sentences from {}'.format(len(file_sents), file_path))

        self.seq_len = model_wrapper.config.get('seq_len', LM_SEQ_LEN)
        self.featurizer = model_wrapper.featurizer
        assert self.featurizer is not None

        print('Fitting featurizer')
        self.featurizer.fit(self.raw_sents)
        print('Found {} tokens'.format(len(self.featurizer.tokenizer.word_counts.keys())))

        print('Tokenizing files')
        raw_data = self.featurizer.transform(self.raw_sents)
        self.raw_data = raw_data
        self.process_raw(batch_size)

    def process_raw(self, batch_size):
        n_batch = self.raw_data.size(0) // batch_size
        batch_data = self.raw_data.narrow(0, 0, n_batch * batch_size)
    
        batch_data = batch_data.view(batch_size, -1).t().contiguous()
        self.batch_data = batch_data
        self.n_batch = n_batch

    def get_save_name(self):
        return 'wikitext-data.bin'

    def save(self):
        torch.save({
            'featurizer': self.featurizer,
            'data': self.raw_data,
            'raw_sents': self.raw_sents
        }, self.get_save_name())
        print('Finished saving preprocessed dataset')

    def load(self, fp, model_wrapper, batch_size=64):
        state = torch.load(fp)
        self.featurizer = state['featurizer']
        model_wrapper.featurizer = state['featurizer']
        self.raw_data = state['data']
        self.seq_len = model_wrapper.config.get('seq_len', LM_SEQ_LEN)
        self.process_raw(batch_size)
        print('Finished loading preprocessed dataset')

    def __len__(self) -> int:
        return self.n_batch

    def __getitem__(self, index) -> Iterable:
        seq_len = min(self.seq_len, len(self.batch_data) - 1 - index)
        X = self.batch_data[index:index+seq_len].long()
        y = self.batch_data[index+1:index+1+seq_len].view(-1)
        return X, y

class LanguageModelLearner(ILearner):

    def __init__(self, model, *args, **kwargs):
        super(LanguageModelLearner, self).__init__(
            model, *args, 
            preprocess_batch=True, 
            auto_optimize=True, **kwargs)

    def on_training_start(self):
        config = self.model_wrapper.config or dict()
        num_words = config.get('num_words', LM_VOCAB_SIZE)
        embedding_dim = config.get('embedding_dim', LM_HIDDEN_DIM)
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
        self.clip_grad = config.get('clip_grad', 5)
        self.alpha = config.get('alpha', 2)
        self.beta = config.get('beta', 1)

    def on_epoch(self, X, y):
        # temporary fix for dataloader
        X = X[0]
        y = y[0]

        if self.hidden is None:
            batch_size = X.size(1)
            self.hidden = self.model_wrapper.model.init_hidden(batch_size)
        else:
            self.hidden = self.model_wrapper.repackage_hidden(self.hidden)
        
        logits, self.hidden, raw_outputs, outputs = \
            self.model_wrapper.model(X, self.hidden, return_raws=True)
        loss = self.criterion(
            self.model_wrapper.model.decoder.weight, 
            self.model_wrapper.model.decoder.bias,
            outputs,
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
            'loss': loss.detach().cpu().item(), 
            'logits': log_probs.cpu().detach()
        }

    def calculate_metrics(self, logits, y):
        return None