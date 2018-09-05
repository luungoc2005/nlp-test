import torch
import numpy as np
import torch.nn as nn
from common.torch_utils import to_gpu
from common.wrappers import ILearner
from common.metrics import accuracy, recall, precision, f1
from sklearn.utils import class_weight
from common.utils import to_categorical

class FastTextLearner(ILearner):

    def __init__(self, model, *args, **kwargs):
        super(FastTextLearner, self).__init__(model, *args, auto_optimize=True, **kwargs)

    def create_ngram_set(self, input_list, ngram_value=2):
        """
        Extract a set of n-grams from a list of integers.
        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
        {(4, 9), (4, 1), (1, 4), (9, 4)}
        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
        [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
        """
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))

    def init_on_data(self, X, y):
        tokens = [self.model_wrapper.tokenize_fn(sent) for sent in X]
        self.model_wrapper.tokenizer.fit_on_texts(tokens)

        if self.model_wrapper._ngrams > 1:
            # print('Adding {}-gram features'.format(ngram_range))
            # Create set of unique n-gram from the training set.
            ngram_set = set()
            for input_list in X:
                for i in range(2, self._model_wrapper._ngrams + 1):
                    set_of_ngram = self.create_ngram_set(input_list, ngram_value=i)
                    ngram_set.update(set_of_ngram)

        start_index = self._model_wrapper.num_words + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}
        self.model_wrapper.token_indice = token_indice
        self.model_wrapper.indice_token = indice_token

        max_features = np.max(list(indice_token.keys())) + 1
        self.model_wrapper._max_features = max_features
        self.model_wrapper._kwargs['config']['max_features'] = max_features

        class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
        self.class_weights = to_gpu(torch.from_numpy(class_weights).float())

        # self.criterion = nn.MultiLabelSoftMarginLoss(weight=class_weights, reduction='sum')
        # self.criterion = nn.NLLLoss(weight=class_weights, reduction='sum')
        # self.criterion = nn.NLLLoss(weight=self.class_weights, reduction='sum')
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

    def on_epoch(self, X, y):
        logits = self.model_wrapper.model(X)
        loss = self.criterion(logits, y)

        loss.backward()

        return {
            'loss': loss.detach().item(), 
            'logits': logits.detach()
        }

    def calculate_metrics(self, logits, y):
        return {
            'accuracy': accuracy(logits, y)
            # 'f1': f1(logits, y),
            # 'precision': precision(logits, y),
            # 'recall': recall(logits, y)
        }