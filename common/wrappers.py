import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import warnings
import pickle
import os
from torch.utils.data import Dataset, DataLoader
from common.torch_utils import set_trainable, children, to_gpu, USE_GPU
from typing import Iterable

class IModel(object):

    def __init__(self, 
        model_class=None,
        config=None, 
        from_fp=None, 
        predict_fn=None, 
        featurizer=None,
        *args, **kwargs):

        self._model_class = model_class
        self._from_fp = from_fp
        self._args = args
        self._kwargs = kwargs or dict()
        self._criterion = None
        self._model = None
        self._predict_fn = predict_fn
        self._featurizer = featurizer
        self.config = config or dict()

    def init_model(self):
        if self._from_fp is None:
            model_state = None
        else:
            self._model = None

            if torch.cuda.is_available():
                model_state = torch.load(self._from_fp)
            else:
                model_state = torch.load(self._from_fp, map_location=lambda storage, loc: storage)

        if model_state is None:
            config = self.config or dict()
        else:
            config = model_state.get('config', dict())
            self.config = config

        if self.is_pytorch_module():
            # re-initialize model with loaded config
            self._model = to_gpu(self._model_class(config=config, *self._args, **self._kwargs))
        else:
            # initialize model normally
            self._model = self._model_class(*self._args, **self._kwargs)

        if model_state is not None:
            featurizer = model_state.get('featurizer', None)

            if featurizer is None:
                if self._featurizer is None:
                    warnings.warn('Featurizer is not found in this binary. This is likely to be an error')
            else:
                # print('Featurizer found: ', featurizer)
                self._featurizer = featurizer
            state_dict = model_state.get('state_dict', None)

            if self.is_pytorch_module() and state_dict is not None:
                self._model.load_state_dict(state_dict)

            self.load_state_dict(model_state)

        self.on_model_init()

    def on_model_init(self): pass

    def preprocess_dataset_X(self, X):
        return X

    def preprocess_dataset_y(self, y):
        return y

    def preprocess_input(self, X):
        return X
    
    def preprocess_output(self, y):
        return y

    def get_state_dict(self): raise NotImplementedError

    def __getstate__(self):
        model_state = None
        try:
            model_state = self.get_state_dict()
        except NotImplementedError:
            model_state = {}
            warnings.warn('get_state_dict() is not implemented.')

        if self.is_pytorch_module():
            model_state['state_dict'] = self._model.state_dict()
        
        if self.config is not None:
            model_state['config'] = self.config
        else:
            model_state['config'] = dict() # default to empty object
        
        if self._featurizer is not None:
            # print('Featurizer found: ', self._featurizer)
            model_state['featurizer'] = self._featurizer
        return model_state

    def load_state_dict(self, state_dict, *args, **kwargs): pass

    def infer_predict(self, logits): return logits

    def is_pytorch_module(self) -> bool: return self._model_class is not None and issubclass(self._model_class, nn.Module)

    def transform(self, X, interpret_fn=None, return_logits=False):
        if self._model is None: return
        is_pytorch = self.is_pytorch_module()

        if is_pytorch:
            self._model.eval()

        with torch.no_grad():
            if self._featurizer is not None:
                X = self._featurizer.transform(X)
            
            X = self.preprocess_input(X)

            if X is None:
                logits = None
            else:
                if self._predict_fn is None:
                    logits = self._model(X)
                else:
                    logits = self._predict_fn(X)
        
        if is_pytorch:
            self._model.train()
        
        if return_logits or logits is None:
            return logits
        elif interpret_fn is not None:
            return interpret_fn(logits)
        else:
            return self.infer_predict(logits)
    
    def freeze_to(self, n):
        c = self.get_layer_groups()
        for l in c:
            set_trainable(l, False)
        for l in c[n:]:
            set_trainable(l, True)

    def freeze_all_but(self, n):
        c = self.get_layer_groups()
        for l in c:
            set_trainable(l, False)
        set_trainable(c[n], True)

    def unfreeze(self): self.freeze_to(0)

    def freeze(self):
        c = self.get_layer_groups()
        for l in c:
            set_trainable(l, False)

    def layers_count(self):
        return len(self.get_layer_groups())

    def get_layer_groups(self):
        # Sample implementation
        # return [
        #     *zip(self.word_encoder.get_layer_groups()),
        #     (self.highway, self.dropout),
        #     self.lstm,
        #     self.hidden2tag
        # ]
        raise NotImplementedError

    
    def loss_function(self, X, y):
        raise NotImplementedError
    
    @property
    def model(self): return self._model

    @property
    def training(self): return self._model.training if hasattr(self._model, 'training') else False

    @model.setter
    def model(self, value): self._model = value

    @property
    def criterion(self): return self._criterion

    @criterion.setter
    def criterion(self, value): self._criterion = value

    @property
    def featurizer(self): return self._featurizer

    @featurizer.setter
    def featurizer(self, value): self._featurizer = value

    def save(self, fp):
        # Exclude functions to avoid pickle failing
        predict_fn = self._predict_fn
        self._predict_fn = None

        torch.save(self.__getstate__(), fp, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    
        self._predict_fn = predict_fn

    def summary(self):
        return self.__str__()

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def __str__(self):
        return self._model.__str__()


class IFeaturizer(object):

    def __init__(self): pass

    def fit(self): pass

    def transform(self): return

    def fit_transform(self): return

    def inverse_transform(self): return

    def get_output_shape(self): return (0,)

class ILearner(object):

    def __init__(self, 
        model_wrapper, 
        data=None, 
        val_data=None, 
        optimizer_fn='adam', 
        optimizer_kwargs={},
        auto_optimize=True,
        preprocess_batch=False,
        uneven_batch_size=False,
        collate_fn=None):
        """
        data: Dataset or tuple (X_train, y_train)
        """
        self._model_wrapper = model_wrapper
        self._optimizer_kwargs = optimizer_kwargs
        self._verbose = 1
        self._optimizer_fn = None
        self._optimizer = None
        self._metrics = None
        self._current_epoch = 0
        self._batch_idx = 0
        self._auto_optimize = auto_optimize
        self._preprocess_batch = preprocess_batch
        self._uneven_batch_size = uneven_batch_size
        self._collate_fn = collate_fn
        self._halt = False # prematurely halt training

        if data is not None:
            self.set_training_data(data)
    
        if optimizer_fn is not None:
            self._optimizer_fn = self.get_optimizer_fn(optimizer_fn)
        else:
            assert self._auto_optimize == False, 'Cannot auto-optimize with None type optimizer function'
            self._optimizer_fn = None

    def get_optimizer_fn(self, optimizer_fn):
        if callable(optimizer_fn):
            return optimizer_fn
        elif isinstance(optimizer_fn, str):
            if optimizer_fn == 'adam':
                return optim.Adam
            elif optimizer_fn == 'rmsprop':
                return optim.RMSprop
            elif optimizer_fn == 'sgd':
                return optim.SGD
            elif optimizer_fn == 'asgd':
                return optim.ASGD
            else:
                raise ValueError('Unsupported optimizer name')
        else:
            raise ValueError('Unsupported optimizer type')

    def init_on_dataset(self, dataset): pass

    def init_on_data(self, X, y): pass

    """
    Fitting tokenizers; initialize params etc. here
    """
    
    def on_training_start(self): pass

    """
    Triggered after model is initialized
    """
    def on_model_init(self): pass
    
    def on_training_end(self): pass

    def on_epoch_start(self): pass

    def on_epoch_end(self): pass
    
    # Runs the forward pass
    # Returns: (loss, logits) loss and raw results of the model
    
    def on_epoch(self, X, y): return

    def calculate_metrics(self, logits, y): pass

    def evaluate_all(self, X):
        """
        Should return predicted targets on the training set
        """
        raise NotImplementedError

    @property
    def optimizer(self): return self._optimizer

    @optimizer.setter
    def optimizer(self, value): self._optimizer = value

    def _convert_to_tuple(self, data):
        assert isinstance(data, list) and len(data) > 0 and isinstance(data[0], tuple)
        X = [item_X for item_X, _ in data]
        y = [item_y for _, item_y in data]
        return X, y

    def set_training_data(self, data):
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], tuple):
            data = self._convert_to_tuple(data)

        assert isinstance(data, tuple) or isinstance(data, Dataset), \
            'data must be either in DataLoader or List format'

        self._is_dataset = isinstance(data, Dataset)

        self._data = data

    def set_validation_data(self, data):
        assert isinstance(data, tuple) or isinstance(data, Dataset), \
            'data must be either in DataLoader or List format'

        self._val_data = data

    def find_lr(self, lr_range, fit_args) -> Iterable:
        losses = []
        for lr in lr_range:
            print('Fitting with optimizer {} lr={}'.format(str(self._optimizer_fn), lr))
            self._optimizer_kwargs = self._optimizer_kwargs or dict()
            self._optimizer_kwargs['lr'] = lr
            self.fit(**fit_args)
            loss = self.metrics['loss']
            losses.append(loss)
            print('Loss: {}'.format(str(loss)))
        return losses

    def fit(self,
        training_data=None,
        validation_data=None,
        epochs=1,
        minibatches=None,
        epoch_start=0, 
        batch_size=64, 
        shuffle=True,
        callbacks=[]):

        if self._uneven_batch_size: batch_size = 1

        if training_data is not None: self.set_training_data(training_data)
        if validation_data is not None: self.set_validation_data(validation_data)

        for callback in callbacks: callback.set_learner(self)

        self._callbacks = callbacks or []
        self._n_epochs = epochs

        # Preprocess data. If data is already a dataset class
        # then preprocessing logic should be implemented in the class
        if not self._is_dataset:
            X, y = self._data

            # Process input and output data - if needed (tokenization etc.)
            if self.model_wrapper._featurizer is not None:
                self.model_wrapper._featurizer.fit(X)
            
            X = self.model_wrapper.preprocess_dataset_X(X)
            y = self.model_wrapper.preprocess_dataset_y(y)

            self.init_on_data(X, y)

            # Preprocess all batches of data (adding n-grams etc.)
            # If data should be lazily processed, use the Dataset class instead.

            if self._preprocess_batch:
                if self.model_wrapper._featurizer is not None:
                    dataset = BatchPreprocessedDataset(X, y,
                        input_process_fn=lambda _X: self.model_wrapper.preprocess_input(
                            self.model_wrapper._featurizer.transform(_X)
                        ),
                        output_process_fn=self.model_wrapper.preprocess_output,
                        batch_size=batch_size)
                else:
                    dataset = BatchPreprocessedDataset(X, y,
                        input_process_fn=self.model_wrapper.preprocess_input,
                        output_process_fn=self.model_wrapper.preprocess_output,
                        batch_size=batch_size)
            else:
                if self.model_wrapper._featurizer is not None:
                    X = self.model_wrapper._featurizer.transform(X)
                    
                X = self.model_wrapper.preprocess_input(X)
                y = self.model_wrapper.preprocess_output(y)

                if not self._uneven_batch_size:
                    dataset = GenericDataset(X, y)
        else:
            dataset = self._data

            self.init_on_dataset(dataset)

        # Call on_training_start hooks
        self.on_training_start()
        for callback in self.callbacks: callback.on_training_start()

        if self._verbose == 2:
            from tqdm import trange
            iterator = trange(epoch_start, self._n_epochs, desc='Epochs', leave=False)
        else:
            iterator = range(epoch_start, self._n_epochs)
        
        cpu_count = int(os.environ.get('NUM_WORKERS', max(mp.cpu_count() - 1, 1)))

        if batch_size is None:
            batch_size = len(dataset)

        if USE_GPU:
            mp.set_start_method('spawn')

        if not self._uneven_batch_size:
            if self._collate_fn is None:
                data_loader = DataLoader(dataset, 
                    batch_size=batch_size, 
                    num_workers=cpu_count,
                    shuffle=shuffle
                )
            else:
                data_loader = DataLoader(dataset, 
                    batch_size=batch_size, 
                    num_workers=cpu_count,
                    shuffle=shuffle,
                    collate_fn=self._collate_fn
                )
        else:
            data_loader = [([X[idx]], [y[idx]]) for idx in range(len(X))]

        if self.model_wrapper._featurizer is not None:
            self.model_wrapper.config['input_shape'] = self.model_wrapper._featurizer.get_output_shape()

        if self.model_wrapper.model is None: self.model_wrapper.init_model()
        self.on_model_init()

        # optimizer must be initialized after the model
        if self.optimizer is None and self._auto_optimize:
            self.optimizer = self._optimizer_fn(
                filter(lambda p: p.requires_grad, self._model_wrapper._model.parameters()),
                **self._optimizer_kwargs
            )

        if self.model_wrapper.is_pytorch_module() and not hasattr(self, 'criterion'):
            raise ValueError('Criterion must be set for the Learner class before training')

        # Main training loop
        try:
            for epoch in iterator:
                if self._halt: # For early stopping
                    self._halt = False
                    break
                
                self._current_epoch = epoch
                self._metrics = None

                self.on_epoch_start()

                for callback in self._callbacks: callback.on_epoch_start()
                
                for batch_idx, (X_batch, y_batch) in enumerate(data_loader, 0):
                    if self._halt: # For early stopping / skipping batches
                        break

                    self._batch_idx = batch_idx

                    for callback in self.callbacks: callback.on_batch_start()

                    # auto_optimize: auto handling the optimizer
                    if self._auto_optimize: self.optimizer.zero_grad()

                    epoch_ret = self.on_epoch(to_gpu(X_batch), to_gpu(y_batch))

                    if epoch_ret is not None:
                        if 'logits' in epoch_ret:
                            with torch.no_grad():
                                batch_metrics = self.calculate_metrics(epoch_ret['logits'].detach(), y_batch) or {}
                        else:
                            batch_metrics = {}

                        if 'loss' in epoch_ret:
                            batch_metrics['loss'] = epoch_ret['loss']
                        
                        self._batch_metrics = batch_metrics

                        if self._metrics is None:
                            self._metrics = batch_metrics
                        else:
                            self._metrics = {k: v + batch_metrics[k] for k, v in self._metrics.items()}

                    if self._auto_optimize: self.optimizer.step()

                    for callback in self.callbacks: callback.on_batch_end()

                    if epochs == 1 and minibatches is not None:
                        if batch_idx >= minibatches:
                            self._halt = True

                self.on_epoch_end()

                for callback in self.callbacks: callback.on_epoch_end()

        except KeyboardInterrupt:
            warnings.warn('Training aborted')

        for callback in self.callbacks: callback.on_training_end()

        self.on_training_end()

    def set_verbosity_level(self, level):
        self._verbose = level

    @property
    def model_wrapper(self): return self._model_wrapper

    @property
    def data(self): return self._data

    @property
    def callbacks(self): return self._callbacks

    @property
    def verbose(self): return self._verbose

    @property
    def metrics(self): 
        if self._metrics is None: return None
        return {k: v / float(self._batch_idx + 1) for k, v in self._metrics.items()}


class GenericDataset(Dataset):

    def __init__(self, X, y):
        self.n_samples = len(X)
        self.samples = X
        self.labels = y

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

class BatchPreprocessedDataset(Dataset):

    def __init__(self, X, y, 
        input_process_fn, 
        output_process_fn, 
        batch_size=64):
        self.n_samples = len(X)

        # preprocess batch
        self._buffer_X = None
        self._buffer_y = None

        for start_idx in range(0, self.n_samples, batch_size):
            batch_X = X[start_idx:start_idx + batch_size]
            batch_y = y[start_idx:start_idx + batch_size]

            batch_X = input_process_fn(batch_X)
            batch_y = output_process_fn(batch_y)

            if self._buffer_X is None:
                # Create new buffers
                if batch_X.ndimension() == 1:
                    self._buffer_X = torch.zeros(self.n_samples)
                else:
                    self._buffer_X = torch.zeros(self.n_samples, *batch_X.size()[1:])
                
                if batch_y.ndimension() == 1:
                    self._buffer_y = torch.zeros(self.n_samples)
                else:
                    self._buffer_y = torch.zeros(self.n_samples, *batch_y.size()[1:])

                # Cast buffer dtype to the same as the first batch
                self._buffer_X.type(batch_X.dtype)
                self._buffer_y.type(batch_y.dtype)

            self._buffer_X[start_idx:start_idx + batch_size] = batch_X
            self._buffer_y[start_idx:start_idx + batch_size] = batch_y

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index) -> Iterable:
        return self._buffer_X[index], self._buffer_y[index]