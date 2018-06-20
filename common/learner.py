import torch
from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader


class IModel(metaclass=ABCMeta):

    @abstractmethod
    def get_state_dict(self): raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict): raise NotImplementedError

    @abstractmethod
    def predict(self, input): raise NotImplementedError

    @property
    def model(self): return self._model

    def save(self, fp):
        torch.save(self.get_state_dict(), fp)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


class ILearner(metaclass=ABCMeta):

    def __init__(self, data, model, callbacks):
        self._data = data
        self._model = model
        self._callbacks = callbacks or []
        self._use_dataloader = False
        self._verbose = 1

        if isinstance(data, DataLoader):
            self._use_dataloader = True
        else:
            assert isinstance(data, tuple), 'data must be either in DataLoader or List format'

        for callback in callbacks: callback.set_learner(self)

    @abstractmethod
    def preprocess_data(self, input):
        pass

    @abstractmethod
    def on_epoch(self, x, y): raise NotImplementedError

    @abstractmethod
    def evaluate_all(self):
        """
        Should return predicted targets on the training set
        """
        raise NotImplementedError

    def fit(self, x, y, n_epochs, epoch_start=0):
        if not self._use_dataloader:
            x = self.preprocess_data(x)

        for callback in self._callbacks: callback.on_training_start()

        if self._verbose == 2:
            from tqdm import trange
            iterator = trange(epoch_start, n_epochs + 1, desc='Epochs', leave=False)
        else:
            iterator = range(epoch_start, n_epochs + 1)

        for _ in iterator:

            for callback in self._callbacks: callback.on_epoch_start()

            self.on_epoch(x, y)

            for callback in self._callbacks: callback.on_epoch_end()

        for callback in self._callbacks: callback.on_training_end()


    @property
    def model(self): return self._model

    @property
    def data(self): return self._data

    @property
    def callbacks(self): return self._callbacks

    @property
    def verbose(self): return self._verbose
