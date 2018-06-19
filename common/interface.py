from abc import ABCMeta, abstractmethod


class IModel(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_model_state(self): raise NotImplementedError


class ILearner(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, input, target): raise NotImplementedError

    @abstractmethod
    def on_epoch(self): raise NotImplementedError


class IDataLoaderLearner(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, dataloader): raise NotImplementedError

    @abstractmethod
    def on_epoch(self): raise NotImplementedError


class ICallback(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def on_training_start(self): raise NotImplementedError

    @abstractmethod
    def on_epoch_start(self): raise NotImplementedError

    @abstractmethod
    def on_epoch_end(self): raise NotImplementedError

    @abstractmethod
    def on_training_end(self): raise NotImplementedError
