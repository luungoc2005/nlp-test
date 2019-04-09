import torch
import numpy as np
import torch.nn as nn
from common.torch_utils import to_gpu
from common.wrappers import ILearner
from common.metrics import accuracy, recall, precision, f1
from sklearn.utils import class_weight
from common.utils import to_categorical
# from common.smooth_topk.svm import SmoothSVM

class FastTextLearner(ILearner):

    def __init__(self, model, *args, **kwargs):
        super(FastTextLearner, self).__init__(model, *args, auto_optimize=False, **kwargs)
        self.emb_train_epochs = self.model_wrapper.config.get('emb_train_epochs', 0)

    def init_on_data(self, X, y):
        self.model_wrapper.label_encoder.fit(y)
        self.model_wrapper.n_classes = len(self.model_wrapper.label_encoder.classes_)
        self.model_wrapper.config['num_classes'] = self.model_wrapper.n_classes

        config = self.model_wrapper.config
        if 'contexts' in config:
            contexts = config['contexts']
            contexts_list = [
                contexts[label] if label in contexts else []
                for label in self.model_wrapper.label_encoder.classes_
            ]
            self.model_wrapper.config['contexts'] = contexts_list
            # print('number of contexts: %s' % str(len(contexts_list)))

        y_labels = self.model_wrapper.label_encoder.transform(y)
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_labels), y_labels)
        self.class_weights = to_gpu(torch.from_numpy(class_weights).float())

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        # self.criterion = SmoothSVM(
        #     self.model_wrapper.config['num_classes'], k=1, alpha=1.
        # )

    def on_epoch(self, X, y):
        logits = self.model_wrapper.model(X)
        loss = self.criterion(logits, y)

        return {
            'logits': logits,
            'loss': loss
        }

    def calculate_metrics(self, logits, y):
        return {
            'accuracy': accuracy(logits, y)
            # TODO: somehow calculating these causes the ipykernel to die
            # 'f1': f1(logits, y),
            # 'precision': precision(logits, y),
            # 'recall': recall(logits, y)
        }