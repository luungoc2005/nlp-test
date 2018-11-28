import time
import warnings
import torch.nn as nn
from common.utils import timeSince, asMinutes
from abc import ABCMeta, abstractmethod
from os import path, remove
from collections import deque
from typing import Union, Iterable, Callable

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
        every_batch:int = 1000,
        every_epoch:int = 1,
        trigger_fn_batch:Callable = None,
        trigger_fn_epoch:Callable = None,
        fn_batch_kwargs:object = {},
        fn_epoch_kwargs:object = {},
        metrics:Iterable[str] = ['loss', 'accuracy']):

        super(PeriodicCallback, self).__init__()
        self.every_epoch = every_epoch
        self.every_batch = every_batch
        self.metrics = metrics
        self.trigger_fn_batch = trigger_fn_batch
        self.trigger_fn_epoch = trigger_fn_epoch
        self.fn_batch_kwargs = fn_batch_kwargs
        self.fn_epoch_kwargs = fn_epoch_kwargs

    def on_training_start(self):
        self.start = time.time()

    def on_batch_end(self):
        if self.every_batch > 0 and self.trigger_fn_batch is not None:
            if ((self._learner._batch_idx + 1) % self.every_batch) == 0:
                self.trigger_fn_batch(**self.fn_batch_kwargs)

    def on_epoch_end(self):
        if self.every_epoch > 0 and self.trigger_fn_epoch is not None:
            if ((self._learner._current_epoch + 1) % self.every_epoch) == 0:
                self.trigger_fn_epoch(**self.fn_epoch_kwargs)

class PrintLoggerCallback(PeriodicCallback):

    def __init__(self, 
        log_every:int = 5, 
        log_every_batch:int = -1, 
        metrics:Iterable[str] = ['loss', 'accuracy'], 
        logging_fn:Callable = print):

        super(PrintLoggerCallback, self).__init__(
            every_batch=log_every_batch,
            every_epoch=log_every,
            metrics=metrics,
            trigger_fn_batch=self.print_line,
            fn_epoch_kwargs={'print_minibatch': True},
            trigger_fn_epoch=self.print_line
        )
        self.log_every = log_every
        self.log_every_batch = log_every_batch
        self.metrics = metrics
        self.logging_fn = logging_fn
        self.last_log_at = 0

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
            print_line += ' - %.2f it/s' % ()
        metrics = self.learner.metrics
        # metrics = self.learner._batch_metrics
        if metrics is not None:
            for key in self.metrics:
                if key in metrics:
                    print_line += ' - %s: %.4f' % (key, metrics[key])

        self.logging_fn(print_line)

class TensorboardCallback(PeriodicCallback):

    def __init__(self, 
        log_every:int = 5, 
        log_every_batch:int = -1, 
        metrics:Iterable[str] = ['loss', 'accuracy']):

        super(TensorboardCallback, self).__init__(
            every_batch=log_every_batch,
            every_epoch=log_every,
            metrics=metrics,
            trigger_fn_batch=self.print_line,
            fn_epoch_kwargs={'print_minibatch': True},
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
    def __init__(self, 
        monitor:str = 'loss', 
        tolerance:float = 1e-6, 
        patience:int = 5, 
        trigger_fn:Callable = None):
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
    def __init__(self, 
        monitor:str = 'loss', 
        tolerance:float = 1e-6, 
        patience:int = 5, 
        logging_fn:Callable = print):
        super(EarlyStoppingCallback, self).__init__(
            monitor=monitor,
            tolerance=tolerance,
            patience=patience,
            trigger_fn=self.stop_training
        )
        assert monitor in ['loss', 'accuracy'], \
            'Early Stopping only implements loss and accuracy metrics at the moment'

        self.logging_fn = print

    def stop_training(self, monitor_val):
        self.logging_fn('Best monitor value `%s` == %4f reached. Early stopping' % (self.monitor, monitor_val))
        self._learner._halt = True

class ReduceLROnPlateau(MetricsTriggeredCallback):
    def __init__(self, 
        monitor:str = 'loss', 
        tolerance:float = 1e-6, 
        patience:int = 1, 
        logging_fn:Callable = print, 
        reduce_factor: Union[int, float] = 5, 
        min_lr:float = 1e-6):
        super(ReduceLROnPlateau, self).__init__(
            monitor=monitor,
            tolerance=tolerance,
            patience=patience,
            trigger_fn=self.reduce_lr
        )
        assert monitor in ['loss', 'accuracy'], \
            'Early Stopping only implements loss and accuracy metrics at the moment'

        self.logging_fn = print
        self.reduce_factor = reduce_factor
        self.min_lr = min_lr

    def reduce_lr(self, monitor_val):
        optimizer = self._learner._optimizer
        current_lr = optimizer.param_groups[0]['lr']
        new_lr = current_lr / self.reduce_factor

        if new_lr > self.min_lr:
            self.logging_fn('Monitor value plateaued at `%s` == %4f. Applying new learning rate: %4f -> %4f' % (self.monitor, monitor_val, current_lr, new_lr))
            self._learner._optimizer.param_groups[0]['lr'] = new_lr
        else:
            self.logging_fn('Minimum learning rate reached. Early stopping')
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