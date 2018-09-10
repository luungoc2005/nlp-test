import torch
import numpy as np
import torch.nn as nn
from common.torch_utils import to_gpu
from common.wrappers import ILearner
from common.metrics import accuracy, recall, precision, f1
from common.langs.vi_VN.utils import remove_tone_marks, random_remove_marks
# import lightgbm as lgb

class EnsembleLearner(ILearner):

    def __init__(self, model, *args, **kwargs):
        super(EnsembleLearner, self).__init__(model, *args, **kwargs, auto_optimize=False, optimizer_fn=None)
        self.criterion = None

    def init_on_data(self, X, y):
        tokens = [self.model_wrapper.tokenize_fn(sent) for sent in X]
        tokens_marks_removed = [
            [remove_tone_marks(token) for token in sent]
            for sent in tokens
        ]
        self.model_wrapper.tokenizer.fit_on_texts(tokens)
        
        self.model_wrapper.label_encoder.fit(y)
        self.n_classes = self.model_wrapper.label_encoder.classes_.shape[0]

    def on_epoch(self, X, y):
        batch_len = X.size(0)
        start_idx = self.buffer_pointer
        self.train_X_buffer[start_idx:start_idx+batch_len] = X.numpy()
        self.train_y_buffer[start_idx:start_idx+batch_len] = y.numpy()
        self.buffer_pointer += batch_len

        return