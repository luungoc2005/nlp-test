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
from typing import Union, Tuple, Iterable

class WikiTextDataset(Dataset):

    def __init__(self, model_wrapper, data_path, batch_size=64):
        super(WikiTextDataset, self).__init__()
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
        batch_data = self.featurizer.transform(self.raw_sents)
        n_batch = len(batch_data) // batch_size
        batch_data = torch.LongTensor(batch_data).narrow(0, 0, n_batch * batch_size)
        batch_data = batch_data.view(batch_size, -1).t().contiguous()
        self.batch_data = batch_data

    def __len__(self) -> int:
        return len(self.raw_sents)

    def __getitem__(self, index) -> Iterable:
        seq_len = min(self.seq_len, len(self.batch_data) - 1 - index)
        X = self.batch_data[i:i+seq_len]
        y = self.batch_data[i+1:i+1+seq_len].view(-1)
        return X, y

class LanguageModelLearner(ILearner):

    def __init__(self, model, *args, **kwargs):
        super(LanguageModelLearner, self).__init__(
            model, *args, 
            preprocess_batch=True, 
            auto_optimize=True, **kwargs)

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
            batch_size = X.size(1)
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