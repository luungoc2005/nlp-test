import torch
import numpy as np
import torch.nn as nn
from common.torch_utils import to_gpu
from common.wrappers import ILearner
from common.metrics import accuracy, recall, precision, f1
from sklearn.utils import class_weight
from common.utils import to_categorical

class LMClassifierLearner(ILearner):

    def __init__(self, model, *args, **kwargs):
        super(LMClassifierLearner, self).__init__(model, preprocess_batch=False, *args, **kwargs)

    def init_on_data(self, X, y):
        self.model_wrapper.label_encoder.fit(y)

        n_classes = len(self.model_wrapper.label_encoder.classes_)
        self.model_wrapper.config['num_classes'] = n_classes

        config = self.model_wrapper.config
        if 'contexts' in config:
            contexts = config['contexts']
            contexts_list = [
                contexts[label] if label in contexts else []
                for label in self.model_wrapper.label_encoder.classes_
            ]
            self.model_wrapper.config['contexts'] = contexts_list

    def on_model_init(self):
        self.criterion = nn.CrossEntropyLoss()

    def on_epoch(self, X, y):
        model = self.model_wrapper.model

        X = to_gpu(self.model_wrapper._featurizer.transform(X))
        with_inhibited, max_pool = model(X)
        loss = self.criterion(with_inhibited, y)- 0.000001 * max_pool.sum()

        return {
            'loss': loss,
            'logits': with_inhibited
        }

    def calculate_metrics(self, logits, y):
        return {
            'accuracy': accuracy(logits, y)
            # 'f1': f1(logits, y),
            # 'precision': precision(logits, y),
            # 'recall': recall(logits, y)
        }