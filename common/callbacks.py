import time
import warnings
import torch.nn as nn
from common.utils import timeSince
from abc import ABCMeta, abstractmethod


class ICallback(object):

    def __init__(self):
        self._learner = None

    def set_learner(self, learner):
        self._learner = learner

    @property
    def learner(self): return self._learner

    def on_training_start(self): pass

    def on_epoch_start(self): pass

    def on_batch_start(self): pass

    def on_batch_end(self): pass

    def on_epoch_end(self): pass

    def on_training_end(self): pass

class PrintLoggerCallback(ICallback):

    def __init__(self, log_every=5, metrics=['loss', 'accuracy']):
        super(PrintLoggerCallback, self).__init__()
        self.log_every = log_every
        self.metrics = metrics

    def on_training_start(self):
        self.start = time.time()

    def on_epoch_end(self):
        if ((self._learner._current_epoch + 1) % self.log_every) == 0:
            progress = float(self._learner._current_epoch + 1) / float(self._learner._n_epochs)
            print_line = '%s (%d %d%%)' % (
                timeSince(self.start, progress), 
                self._learner._current_epoch + 1,
                progress * 100
            )
            metrics = self.learner.metrics
            # metrics = self.learner._batch_metrics
            if metrics is not None:
                for key in self.metrics:
                    if key in metrics:
                        print_line += ' - %s: %.4f' % (key, metrics[key])
                
            print(print_line)

class EarlyStoppingCallback(ICallback):
    def __init__(self, monitor='loss', tolerance=1e-6, patience=5):
        super(EarlyStoppingCallback, self).__init__()
        assert monitor in ['loss', 'accuracy'], \
            'Early Stopping only implements loss and accuracy metrics at the moment'
        self.monitor = monitor
        self.tolerance = tolerance
        self.patience = patience

        self.multiplier = 1
        if self.monitor == 'accuracy':
            self.multiplier = -1 # Reverse the monitor

    def on_training_start(self):
        self.wait = 0
        self.best_val = 1e15

    def on_epoch_end(self):
        if self._learner._batch_metrics is None:
            warnings.warn('The Learner class does not return any batch metrics. Early Stopping cannot work here')
            return
        
        if self.monitor not in self._learner._batch_metrics:
            warnings.warn('The Learner class does not return the specified metrics. Early Stopping cannot work here')
            return

        metrics = self.learner.metrics
        # metrics = self.learner._batch_metrics
        monitor_val = metrics[self.monitor] * self.multiplier

        if monitor_val < self.best_val - self.tolerance:
            self.best_val = monitor_val
            self.wait = 1
        else:
            if self.wait >= self.patience:
                print('Best monitor value `%s` == %4f reached. Early stopping' % (self.monitor, monitor_val))
                self._learner._halt = True
            self.wait += 1

class TemperatureScalingCallback(ICallback):

    def __init__(self):
        super(TemperatureScalingCallback, self).__init__()

    def on_training_end(self):
        model = self.learner.model_wrapper.model
