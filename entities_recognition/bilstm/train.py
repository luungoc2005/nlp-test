import torch
import numpy as np
import torch.nn as nn
from common.torch_utils import to_gpu
from common.wrappers import ILearner
from common.metrics import accuracy, recall, precision, f1

class SequenceTaggerLearner(ILearner):

    def __init__(self, model, *args, **kwargs):
        super(SequenceTaggerLearner, self).__init__(
            model, *args, uneven_batch_size=True, **kwargs)

    def on_model_init(self):
        self.criterion = self.model_wrapper.model.neg_log_likelihood

    def on_epoch(self, X, y):
        logits, loss = self.criterion(X, y)

        return {
            'loss': loss,
            'logits': logits.detach()
        }

    def calculate_metrics(self, logits, y):
        with torch.no_grad():
            _, preds = self.model_wrapper.model._viterbi_decode(logits)
            preds = torch.LongTensor(preds)
            
        return {
            'accuracy': accuracy(preds, y[0])
            # 'f1': f1(logits, y),
            # 'precision': precision(logits, y),
            # 'recall': recall(logits, y)
        }