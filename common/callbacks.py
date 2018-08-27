import time
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
            # metrics = self.learner.metrics
            metrics = self.learner._batch_metrics
            if metrics is not None:
                for key in self.metrics:
                    if key in metrics:
                        print_line += ' - %s: %.4f' % (key, metrics[key])
                
            print(print_line)