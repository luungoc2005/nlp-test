import torch
import numpy as np
import torch.nn as nn
from common.torch_utils import to_gpu
from common.wrappers import ILearner
from common.metrics import accuracy, recall, precision, f1
from sklearn.utils import class_weight
from common.utils import to_categorical
from config import EMBEDDING_DIM
# import lightgbm as lgb

class EnsembleLearner(ILearner):

    def __init__(self, model, *args, **kwargs):
        super(EnsembleLearner, self).__init__(model, *args, **kwargs, auto_optimize=False, optimizer_fn=None)
        self.criterion = None

    def init_on_data(self, X, y):
        tokens = [self.model_wrapper.tokenize_fn(sent) for sent in X]
        self.model_wrapper.tokenizer.fit_on_texts(tokens)
        self.n_samples = len(tokens)
        self.train_X_buffer = np.zeros((self.n_samples, EMBEDDING_DIM))
        self.train_y_buffer = np.zeros((self.n_samples,))
        self.buffer_pointer = 0

        self.model_wrapper.label_encoder.fit(y)
        self.n_classes = self.model_wrapper.label_encoder.classes_.shape[0]

        # self.class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
        # self.model_wrapper._kwargs['class_weight'] = self.class_weights

    def on_training_end(self):
        self.model_wrapper.model.fit(self.train_X_buffer, self.train_y_buffer)
        self.model_wrapper._predict_fn = self.model_wrapper.model.predict_proba

        print('Model score: %s' % self.model_wrapper.model.score(self.train_X_buffer, self.train_y_buffer))
        
        # LightGBM
        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'multiclass',
        #     'metric': 'multi_logloss',
        #     'num_class': self.n_classes
        # }
        # dataset = lgb.Dataset(self.train_X_buffer, label=self.train_y_buffer)
        # self.model_wrapper.model = lgb.train(params, dataset)
        # self.model_wrapper._predict_fn = self.model_wrapper.model.predict

    def on_epoch(self, X, y):
        batch_len = X.size(0)
        start_idx = self.buffer_pointer
        self.train_X_buffer[start_idx:start_idx+batch_len] = X.numpy()
        self.train_y_buffer[start_idx:start_idx+batch_len] = y.numpy()
        self.buffer_pointer += batch_len

        return