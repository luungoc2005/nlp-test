import torch
import numpy as np
import torch.nn as nn
from common.torch_utils import to_gpu
from common.wrappers import ILearner
from common.metrics import accuracy, recall, precision, f1

class SequenceTaggerLearner(ILearner):

    def __init__(self, model, *args, **kwargs):
        super(SequenceTaggerLearner, self).__init__(model, *args, auto_optimize=True, **kwargs)
        self.criterion = model.neg_log_likelihood

    def on_epoch(self, X, y):
        # logits = self.model_wrapper.model(X)
        logits, loss = self.criterion(sentence, y)

        loss.backward()

        return {
            'loss': loss.detach().item(),
            'logits': logits.detach().item()
        }

    def calculate_metrics(self, logits, y):
        with torch.no_grad():
            _, preds = self.model_wrapper.model._viterbi_decode(logits)
            
        return {
            'accuracy': accuracy(preds, y)
            # 'f1': f1(logits, y),
            # 'precision': precision(logits, y),
            # 'recall': recall(logits, y)
        }