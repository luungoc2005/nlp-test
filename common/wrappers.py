import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from common.torch_utils import set_trainable, children, to_gpu

class IModel(object):

    def __init__(self, model_class=None, from_fp=None, predict_fn=None, *args, **kwargs):
        self._model_class = model_class
        self._from_fp = from_fp
        self._args = args
        self._kwargs = kwargs
        self._criterion = None
        self._model = None
        self._predict_fn = predict_fn

    def init_model(self):
        if self._from_fp is None:
            self._model = to_gpu(self._model_class(*self._args, **self._kwargs))
        else:
            self._model = None

            if torch.cuda.is_available():
                self.load_state_dict(torch.load(self._from_fp))
            else:
                self.load_state_dict(torch.load(self._from_fp, map_location=lambda storage, loc: storage))
        
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

    def load_state_dict(self, state_dict, *args, **kwargs): raise NotImplementedError

    def infer_predict(self, logits): return logits

    def is_pytorch_module(self): return self._model is not None and isinstance(self._model, nn.Module)

    def predict(self, X):
        if self._model is None: return
        is_pytorch = self.is_pytorch_module()

        if is_pytorch:
            self._model.eval()

        with torch.no_grad():
            X = self.preprocess_input(X)
            if self._predict_fn is None:
                logits = self._model(X)
            else:
                logits = self._predict_fn(X)
        
        if is_pytorch:
            self._model.train()
        
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

    @model.setter
    def model(self, value): self._model = value

    @property
    def criterion(self): return self._criterion

    @criterion.setter
    def criterion(self, value): self._criterion = value

    def save(self, fp):
        torch.save(self.get_state_dict(), fp)
    
    def summary(self):
        return self.__str__()

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def __str__(self):
        return self._model.__str__()


class ILearner(object):

    def __init__(self, 
        model_wrapper, 
        data=None, 
        val_data=None, 
        optimizer_fn='adam', 
        optimizer_kwargs={},
        auto_optimize=True,
        preprocess_batch=False,
        uneven_batch_size=False):
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
        self._halt = False # prematurely halt training

        if data is not None:
            self.set_training_data(data)
    
        if optimizer_fn is not None:
            if callable(optimizer_fn):
                self._optimizer_fn = optimizer_fn
            elif isinstance(optimizer_fn, str):
                if optimizer_fn == 'adam':
                    self._optimizer_fn = optim.Adam
                elif optimizer_fn == 'rmsprop':
                    self._optimizer_fn = optim.RMSprop
                elif optimizer_fn == 'sgd':
                    self._optimizer_fn = optim.SGD
                else:
                    raise ValueError('Unsupported optimizer name')
            else:
                raise ValueError('Unsupported optimizer type')
        else:
            assert self._auto_optimize == False, 'Cannot auto-optimize with None type optimizer function'
            self._optimizer_fn = None

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

    # Runs the forward pass
    # Returns: (loss, logits) loss and raw results of the model
    
    def on_epoch(self, X, y): pass

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

    def fit(self,
        training_data=None,
        validation_data=None,
        epochs=1, 
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
            X = self.model_wrapper.preprocess_dataset_X(X)
            y = self.model_wrapper.preprocess_dataset_y(y)

            self.init_on_data(X, y)

            # Preprocess all batches of data (adding n-grams etc.)
            # If data should be lazily processed, use the Dataset class instead.

            if self._preprocess_batch:
                dataset = BatchPreprocessedDataset(X, y,
                    input_process_fn=self.model_wrapper.preprocess_input,
                    output_process_fn=self.model_wrapper.preprocess_output,
                    batch_size=batch_size)
            else:
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
        
        cpu_count = max(mp.cpu_count() - 1, 1)

        if batch_size is None:
            batch_size = len(dataset)

        if not self._uneven_batch_size:
            data_loader = DataLoader(dataset, 
                batch_size=batch_size, 
                num_workers=cpu_count,
                shuffle=shuffle
            )
        else:
            data_loader = [([X[idx]], [y[idx]]) for idx in range(len(X))]

        if self.model_wrapper.model is None: self.model_wrapper.init_model()
        self.on_model_init()

        # optimizer must be initialized after the model
        if self.optimizer is None and self._auto_optimize:
            self.optimizer = self._optimizer_fn(
                filter(lambda p: p.requires_grad, self._model_wrapper._model.parameters()),
                **self._optimizer_kwargs
            )

        if not hasattr(self, 'criterion'):
            raise ValueError('Criterion must be set for the Learner class before training')

        # Main training loop
        for epoch in iterator:
            if self._halt: # For early stopping
                self._halt = False
                break
            
            self._current_epoch = epoch
            self._metrics = None

            for callback in self._callbacks: callback.on_epoch_start()
            
            for batch_idx, (X_batch, y_batch) in enumerate(data_loader, 0):
                if self._halt: # For early stopping / skipping batches
                    self._halt = False
                    break

                self._batch_idx = batch_idx

                for callback in self.callbacks: callback.on_batch_start()

                # auto_optimize: auto handling the optimizer
                if self._auto_optimize: self.optimizer.zero_grad()

                epoch_ret = self.on_epoch(X_batch, y_batch)

                if epoch_ret is not None and 'logits' in epoch_ret:
                    batch_metrics = self.calculate_metrics(epoch_ret['logits'], y_batch) or {}
                    
                    if 'loss' in epoch_ret:
                        batch_metrics['loss'] = epoch_ret['loss']
                    
                    self._batch_metrics = batch_metrics

                    if self._metrics is None:
                        self._metrics = batch_metrics
                    else:
                        self._metrics = {k: v + batch_metrics[k] for k, v in self._metrics.items()}

                if self._auto_optimize: self.optimizer.step()

                for callback in self.callbacks: callback.on_batch_end()

            for callback in self.callbacks: callback.on_epoch_end()

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

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self._buffer_X[index], self._buffer_y[index]