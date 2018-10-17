import time
import warnings
import torch.nn as nn
from common.utils import timeSince, asMinutes
from abc import ABCMeta, abstractmethod
from os import path, remove
from collections import deque

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

    def __init__(self, log_every=5, log_every_batch=-1, metrics=['loss', 'accuracy'], logging_fn=print):
        super(PrintLoggerCallback, self).__init__()
        self.log_every = log_every
        self.log_every_batch = log_every_batch
        self.metrics = metrics
        self.logging_fn = logging_fn

    def on_training_start(self):
        self.start = time.time()

    def print_line(self, print_minibatch=False):
        if print_minibatch == False:
            progress = float(self._learner._current_epoch + 1) / float(self._learner._n_epochs)
            print_line = '%s (%d %d%%)' % (
                timeSince(self.start, progress), 
                self._learner._current_epoch + 1,
                progress * 100
            )
        else:
            progress = float(self._learner._current_epoch + 1) / float(self._learner._n_epochs)
            print_line = '%s (%d-%d %d%%)' % (
                timeSince(self.start, progress),
                self._learner._batch_idx + 1,
                self._learner._current_epoch + 1,
                progress * 100
            )
        metrics = self.learner.metrics
        # metrics = self.learner._batch_metrics
        if metrics is not None:
            for key in self.metrics:
                if key in metrics:
                    print_line += ' - %s: %.4f' % (key, metrics[key])

        self.logging_fn(print_line)

    def on_epoch_end(self):
        if self.log_every > 0:
            if ((self._learner._current_epoch + 1) % self.log_every) == 0:
                self.print_line()

    def on_batch_end(self):
        if self.log_every_batch > 0:
            if ((self._learner._batch_idx + 1) % self.log_every_batch) == 0:
                self.print_line(True)

class EarlyStoppingCallback(ICallback):
    def __init__(self, monitor='loss', tolerance=1e-6, patience=5, logging_fn=print):
        super(EarlyStoppingCallback, self).__init__()
        assert monitor in ['loss', 'accuracy'], \
            'Early Stopping only implements loss and accuracy metrics at the moment'
        self.monitor = monitor
        self.tolerance = tolerance
        self.patience = patience
        self.logging_fn = print

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
                self.logging_fn('Best monitor value `%s` == %4f reached. Early stopping' % (self.monitor, monitor_val))
                self._learner._halt = True
            self.wait += 1

class ModelCheckpointCallback(ICallback):

    def __init__(self, 
        every_batch=10,
        every_epochs=1,
        save_last=10,
        logging_fn=print,
        metrics=['loss', 'accuracy']):

        super(ModelCheckpointCallback, self).__init__()
        self.every_epochs = every_epochs
        self.every_batch = every_batch
        self.metrics = metrics
        self.logging_fn = logging_fn
        self.save_last = save_last
        self.file_queue = deque()

    def on_training_start(self):
        self.start = time.time()

    def get_savefile_name(self):
        now = time.time()
        s = now - self.start

        file_name = '{} - epoch {}:{} checkpoint'.format(
            asMinutes(s),
            self._learner._current_epoch + 1,
            self._learner._batch_idx + 1
        )

        metrics = self.learner.metrics
        # metrics = self.learner._batch_metrics
        if metrics is not None:
            for key in self.metrics:
                if key in metrics:
                    file_name += ' - %s: %.4f' % (key, metrics[key])
        file_name += '.bin'
        return file_name

    def save_checkpoint(self):
        new_file_name = self.get_savefile_name()
        self.file_queue.append(new_file_name)

        if self.save_last > 0:
            if len(self.file_queue) > self.save_last:
                old_file_name = self.file_queue.popleft()
                if old_file_name != '' and path.exists(old_file_name) and path.isfile(old_file_name):
                    remove(old_file_name)
                    self.logging_fn('Model Checkpoint: Removing old checkpoint: {}'.format(old_file_name))

        self._learner.save(new_file_name)
        self.logging_fn('Model Checkpoint: Saving checkpoint: {}'.format(new_file_name))

    def on_batch_end(self):
        if self.every_batch > 0:
            if ((self._learner._batch_idx + 1) % self.every_batch) == 0:
                self.save_checkpoint()

    def on_epoch_end(self):
        if self.every_epochs > 0:
            if ((self._learner._current_epoch + 1) % self.every_epochs) == 0:
                self.save_checkpoint()



class TemperatureScalingCallback(ICallback):

    def __init__(self):
        super(TemperatureScalingCallback, self).__init__()

    def on_training_end(self):
        model = self.learner.model_wrapper.model
