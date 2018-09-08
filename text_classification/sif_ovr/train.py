import torch
import numpy as np
import torch.nn as nn
from common.torch_utils import to_gpu
from common.wrappers import ILearner
from common.metrics import accuracy, recall, precision, f1
from sklearn.utils import class_weight
from common.utils import to_categorical
from config import EMBEDDING_DIM
import lightgbm as lgb

class OvrClassifierLearner(ILearner):

    def __init__(self, model, *args, **kwargs):
        super(OvrClassifierLearner, self).__init__(model, preprocess_batch=True, *args, **kwargs, auto_optimize=False, optimizer_fn=None)

    def init_on_data(self, X, y):
        tokens = [self.model_wrapper.tokenize_fn(sent) for sent in X]
        self.model_wrapper.tokenizer.fit_on_texts(tokens)
        self.n_samples = len(tokens)
        self.n_classes = len(np.unique(y))
        self.buffer_pointer = 0
        
        self.model_wrapper.label_encoder.fit(y)
        self.n_classes = self.model_wrapper.label_encoder.classes_.shape[0]
        # self.class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)

    def on_model_init(self):
        self.criterion = self.model_wrapper.model.binary_crossentropy_loss

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