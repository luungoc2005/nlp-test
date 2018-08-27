import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from common.torch_utils import set_trainable, children, to_gpu

class IModel(object):

    def __init__(self, model_class=None, from_fp=None, *args, **kwargs):
        self._model_class = model_class
        self._from_fp = from_fp
        self._args = args
        self._kwargs = kwargs
        self._criterion = None
        self._model = None

    def init_model(self):
        if self._from_fp is None:
            self._model = to_gpu(self._model_class(*self._args, **self._kwargs))
        else:
            self._model = None

            if torch.cuda.is_available():
                self.load_state_dict(torch.load(self._from_fp))
            else:
                self.load_state_dict(torch.load(self._from_fp), map_location=lambda storage, loc: storage)


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

    def predict(self, X):
        self._model.eval()

        with torch.no_grad():
            X = self.preprocess_input(X)
            logits = self._model(X)
        
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
        auto_optimize=True):
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

        if data is not None:
            self.set_training_data(data)
    
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

    def init_on_dataset(self, dataset): pass

    def init_on_data(self, X, y): pass

    """
    Fitting tokenizers; initialize params etc. here
    """
    
    def on_training_start(self): pass
    
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

    def set_training_data(self, data):
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
        epochs=50, 
        epoch_start=0, 
        batch_size=64, shuffle=True,
        callbacks=[]):

        if self.model_wrapper.model is None: self.model_wrapper.init_model()

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

            X = self.model_wrapper.preprocess_input(X)
            y = self.model_wrapper.preprocess_output(y)

            dataset = GenericDataset(X, y)
        else:
            dataset = self._data

            self.init_on_dataset(dataset)

        # Call on_training_start hooks
        self.on_training_start()
        for callback in self.callbacks: callback.on_training_start()

        if self._verbose == 2:
            from tqdm import trange
            iterator = trange(epoch_start, self._n_epochs + 1, desc='Epochs', leave=False)
        else:
            iterator = range(epoch_start, self._n_epochs + 1)
        
        cpu_count = max(mp.cpu_count() - 1, 1)

        data_loader = DataLoader(dataset, 
            batch_size=batch_size, 
            num_workers=cpu_count,
            shuffle=shuffle
        )

        if self.optimizer is None:
            self.optimizer = self._optimizer_fn(
                filter(lambda p: p.requires_grad, self._model_wrapper._model.parameters()),
                **self._optimizer_kwargs
            )

        if self.criterion is None:
            raise ValueError('Criterion must be set for the Learner class before training')

        # Main training loop
        for epoch in iterator:
            self._current_epoch = epoch

            for callback in self._callbacks: callback.on_epoch_start()

            for batch_idx, (X_batch, y_batch) in enumerate(data_loader, 0):
                self._metrics = None
                self._batch_idx = batch_idx

                for callback in self.callbacks: callback.on_batch_start()
                
                # auto_optimize: auto handling the optimizer
                if self._auto_optimize: self.optimizer.zero_grad()

                epoch_ret = self.on_epoch(X_batch, y_batch)

                if 'logits' in epoch_ret:
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

        assert self.n_samples == len(y)

        self.samples = X
        self.labels = y

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]