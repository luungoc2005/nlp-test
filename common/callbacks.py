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

class PeriodicCallback(ICallback):

    def __init__(self, 
        every_batch=1000,
        every_epoch=1,
        trigger_fn_batch=None,
        trigger_fn_epoch=None,
        fn_batch_args={},
        fn_epoch_args={},
        metrics=['loss', 'accuracy']):

        super(PeriodicCallback, self).__init__()
        self.every_epoch = every_epoch
        self.every_batch = every_batch
        self.metrics = metrics
        self.trigger_fn_batch = trigger_fn_batch
        self.trigger_fn_epoch = trigger_fn_epoch
        self.fn_batch_kargs={}
        self.fn_epoch_kargs={}

    def on_training_start(self):
        self.start = time.time()

    def on_batch_end(self):
        if self.every_batch > 0 and self.trigger_fn_batch is not None:
            if ((self._learner._batch_idx + 1) % self.every_batch) == 0:
                self.trigger_fn_batch(**self.fn_batch_kargs)

    def on_epoch_end(self):
        if self.every_epoch > 0 and self.trigger_fn_epoch is not None:
            if ((self._learner._current_epoch + 1) % self.every_epoch) == 0:
                self.trigger_fn_epoch(**self.fn_epoch_kargs)

class PrintLoggerCallback(PeriodicCallback):

    def __init__(self, log_every=5, log_every_batch=-1, metrics=['loss', 'accuracy'], logging_fn=print):
        super(PrintLoggerCallback, self).__init__(
            every_batch=log_every_batch,
            every_epoch=log_every,
            metrics=metrics,
            trigger_fn_batch=self.print_line,
            fn_batch_args={'print_minibatch': True},
            trigger_fn_epoch=self.print_line
        )
        self.log_every = log_every
        self.log_every_batch = log_every_batch
        self.metrics = metrics
        self.logging_fn = logging_fn

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


class TensorboardCallback(PeriodicCallback):

    def __init__(self, log_every=5, log_every_batch=-1, metrics=['loss', 'accuracy']):
        super(TensorboardCallback, self).__init__(
            every_batch=log_every_batch,
            every_epoch=log_every,
            metrics=metrics,
            trigger_fn_batch=self.print_line,
            fn_batch_kargs={'print_minibatch': True},
            trigger_fn_epoch=self.print_line
        )
        self.log_every = log_every
        self.log_every_batch = log_every_batch
        self.metrics = metrics
        
        self.counter = 1

    def on_training_start(self):
        super(TensorboardCallback, self).on_training_start()

        from tensorboardX import SummaryWriter
        self.class_name = self.learner.model_wrapper.__class__.__name__
        self.writer = SummaryWriter(comment='_' + self.class_name)

    def print_line(self, print_minibatch=False):
        self.counter += 1
        metrics = self.learner.metrics
        # metrics = self.learner._batch_metrics
        if metrics is not None:
            for key in self.metrics:
                if key in metrics:
                    # print_line += ' - %s: %.4f' % (key, metrics[key])
                    self.writer.add_scalar(
                        '{}/{}'.format(self.class_name, key),
                        metrics[key],
                        self.counter
                    )

class MetricsTriggeredCallback(ICallback):
    def __init__(self, monitor='loss', tolerance=1e-6, patience=5, trigger_fn=None):
        super(MetricsTriggeredCallback, self).__init__()

        self.monitor = monitor
        self.tolerance = tolerance
        self.patience = patience
        self.trigger_fn = trigger_fn

        self.multiplier = 1
        if self.monitor == 'accuracy':
            self.multiplier = -1 # Reverse the monitor

    def on_training_start(self):
        self.wait = 0
        self.best_val = 1e15

    def on_epoch_end(self):
        if self.trigger_fn is None: return
        
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
                self.trigger_fn(monitor_val)
            self.wait += 1

class EarlyStoppingCallback(MetricsTriggeredCallback):
    def __init__(self, monitor='loss', tolerance=1e-6, patience=5, logging_fn=print):
        super(EarlyStoppingCallback, self).__init__(
            monitor=monitor,
            tolerance=tolerance,
            patience=patience,
            trigger_fn=self.stop_training
        )
        assert monitor in ['loss', 'accuracy'], \
            'Early Stopping only implements loss and accuracy metrics at the moment'
        self.monitor = monitor
        self.tolerance = tolerance
        self.patience = patience
        self.logging_fn = print

        self.multiplier = 1
        if self.monitor == 'accuracy':
            self.multiplier = -1 # Reverse the monitor

    def stop_training(self, monitor_val):
        self.logging_fn('Best monitor value `%s` == %4f reached. Early stopping' % (self.monitor, monitor_val))
        self._learner._halt = True

class ModelCheckpointCallback(PeriodicCallback):

    def __init__(self, 
        every_batch=10000,
        every_epoch=1,
        save_last=10,
        logging_fn=print,
        metrics=['loss', 'accuracy']):

        super(ModelCheckpointCallback, self).__init__(
            every_batch=every_batch,
            every_epoch=every_epoch,
            metrics=metrics,
            trigger_fn_batch=self.save_checkpoint,
            trigger_fn_epoch=self.save_checkpoint
        )
        self.logging_fn = logging_fn
        self.save_last = save_last
        self.file_queue = deque()

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

        self.learner.model_wrapper.save(new_file_name)
        self.logging_fn('Model Checkpoint: Saving checkpoint: {}'.format(new_file_name))


class TemperatureScalingCallback(ICallback):

    def __init__(self):
        super(TemperatureScalingCallback, self).__init__()

    def on_training_end(self):
        # model = self.learner.model_wrapper.model
        return