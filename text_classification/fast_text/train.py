import torch
import numpy as np
import torch.nn as nn
from common.torch_utils import to_gpu
from common.wrappers import ILearner
from common.metrics import accuracy, recall, precision, f1
from sklearn.utils import class_weight
from common.utils import to_categorical

from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
# from common.smooth_topk.svm import SmoothSVM

class FastTextLearner(ILearner):

    def __init__(self, model, *args, **kwargs):
        super(FastTextLearner, self).__init__(model, *args, auto_optimize=False, **kwargs)
        self.emb_train_epochs = self.model_wrapper.config.get('emb_train_epochs', 0)

    def init_on_data(self, X, y):
        self.model_wrapper.label_encoder.fit(y)
        self.model_wrapper.config['num_classes'] = self.model_wrapper.label_encoder.classes_.shape[0]

        y_labels = self.model_wrapper.label_encoder.transform(y)
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_labels), y_labels)
        self.class_weights = to_gpu(torch.from_numpy(class_weights).float())

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        # self.criterion = SmoothSVM(
        #     self.model_wrapper.config['num_classes'], k=1, alpha=1.
        # )

    def on_training_start(self):
        self.start_finetune=False

    def on_epoch_start(self):
        total_epochs = self._n_epochs
        current_epoch = self._current_epoch

        if current_epoch > total_epochs * self.emb_train_epochs:
            if not self.start_finetune:
                print('Switching to pyro')
                self.start_finetune = True
                self.optimizer = Adam({"lr": 1e-3})
                self.svi = SVI(self.model_wrapper.pyro_model, self.model_wrapper.pyro_guide, self.optimizer, loss=Trace_ELBO())
        else:
            if self.optimizer is None:
                self.optimizer = torch.optim.Adam(self.model_wrapper.model.parameters())

    def on_epoch(self, X, y):
        total_epochs = self._n_epochs
        current_epoch = self._current_epoch

        if not self.start_finetune:
            logits = self.model_wrapper.model(X)
            loss = self.criterion(logits, y)

            loss.backward()

            # optimize
            self.optimizer.step()
            self.model_wrapper.model.zero_grad()

            return {
                'loss': loss.detach().item()
            }
        else:
            loss = self.svi.step(X, y)
            return {
                'loss': loss
            }

    def calculate_metrics(self, logits, y):
        return {
            # 'accuracy': accuracy(logits, y)
            # TODO: somehow calculating these causes the ipykernel to die
            # 'f1': f1(logits, y),
            # 'precision': precision(logits, y),
            # 'recall': recall(logits, y)
        }