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

    def init_on_data(self, X, y):
        self.model_wrapper.label_encoder.fit(y)
        self.model_wrapper.config['num_classes'] = self.model_wrapper.label_encoder.classes_.shape[0]

        y_labels = self.model_wrapper.label_encoder.transform(y)
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_labels), y_labels)
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
            # TODO: somehow calculating these causes the ipykernel to die
            # 'f1': f1(logits, y),
            # 'precision': precision(logits, y),
            # 'recall': recall(logits, y)
        }