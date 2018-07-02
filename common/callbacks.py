from abc import ABCMeta, abstractmethod


class ICallback(metaclass=ABCMeta):

    def __init__(self):
        self._learner = None

    def set_learner(self, learner):
        self._learner = learner

    @property
    def learner(self): return self._learner

    @abstractmethod
    def on_training_start(self): pass

    @abstractmethod
    def on_epoch_start(self): pass

    @abstractmethod
    def on_epoch_end(self): pass

    @abstractmethod
    def on_training_end(self): pass

