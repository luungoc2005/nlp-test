import torch
import numpy as np
import torch.nn as nn
from common.torch_utils import to_gpu
from common.wrappers import ILearner
from common.metrics import accuracy, recall, precision, f1
from entities_recognition.transformer.data import collate_transformer_entities_target

class TransformerSequenceTaggerLearner(ILearner):

    def __init__(self, model, *args, **kwargs):
        super(TransformerSequenceTaggerLearner, self).__init__(
            model, 
            *args,
            collate_fn=collate_transformer_entities_target, 
            **kwargs
        )

    def on_model_init(self):
        if self.model_wrapper.model.use_crf:
            self.criterion = self.model_wrapper.model.crf
        else:
            self.criterion = nn.CrossEntropyLoss()

    def on_epoch(self, X, y):
        tags_output, _, _ = self.model_wrapper.model(X)
        if self.model_wrapper.model.use_crf:
            loss = -1 * self.criterion(tags_output, y, reduce=False)
            loss = loss.mean()
        else:
            loss = self.criterion(tags_output.view(-1, tags_output.size(-1)), y.view(-1))

        return {
            'loss': loss,
            'logits': tags_output.detach()
        }

    def calculate_metrics(self, logits, y):
        return {
            'accuracy': accuracy(torch.max(logits, -1)[1], y)
            # 'f1': f1(logits, y),
            # 'precision': precision(logits, y),
            # 'recall': recall(logits, y)
        }