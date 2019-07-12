import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import warnings
import pickle
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from common.utils import dotdict
from common.torch_utils import set_trainable, children, to_gpu, USE_GPU, copy_optimizer_params_to_model, set_optimizer_params_grad, use_data_parallel
from common.quantize import quantize_model, dequantize_model
from typing import Iterable, Union, Callable, Tuple
import inspect

class IFeaturizer(object):

    def __init__(self): pass

    def fit(self): pass

    def transform(self): return

    def fit_transform(self): return

    def inverse_transform(self): return

    def get_output_shape(self) -> Tuple[int]: return (0,)

class IModel(object):

    def __init__(self, 
        model_class:nn.Module = None,
        config:dict = None, 
        from_fp:str = None, 
        predict_fn:Callable = None, 
        featurizer:IFeaturizer = None,
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
        self._quantized = False
        self._onnx = None
        self._onnx_model = None
        self._use_data_parallel = use_data_parallel()

    def init_model(self, update_configs: dict = {}):
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
            self._onnx = model_state.get('onnx', None)

        # convert to dotdict
        if config is dict and not isinstance(config, dotdict):
            config = dotdict(config)
            self.config = config

        config.update(update_configs)

        if self.is_pytorch_module():
            # re-initialize model with loaded config
            self._model = self._model_class(config=config, *self._args, **self._kwargs)
            if self._use_data_parallel:
                self._model = nn.DataParallel(self._model, dim=1)
            # if fp16: self._model.half()
            self._model = to_gpu(self._model)
        else:
            # initialize model normally
            if self._onnx is None:
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

            if self.is_pytorch_module():
                if state_dict is not None:
                    self._model.load_state_dict(state_dict, strict=False)
            elif self._onnx is not None:
                import onnx
                self._onnx_model = onnx.load(self._onnx)
                print('Loaded ONNX model')

            self.load_state_dict(model_state)

        self.config = config
        self.on_model_init()

    def on_model_init(self): pass

    def preprocess_dataset_X(self, X: Union[Iterable, torch.Tensor]):
        return X

    def preprocess_dataset_y(self, y: Union[Iterable, torch.Tensor]):
        return y

    def preprocess_input(self, X: Union[Iterable, torch.Tensor]):
        return X
    
    def preprocess_output(self, y: Union[Iterable, torch.Tensor]):
        return y

    def get_state_dict(self): raise NotImplementedError

    def __getstate__(self) -> dict:
        model_state = {}
        if self.is_pytorch_module():
            if self._model is not None:
                model_state['state_dict'] = self._model.state_dict()
        else:
            model_state['onnx'] = self._onnx

        if self.config is not None:
            model_state['config'] = self.config
        else:
            model_state['config'] = dict() # default to empty object
        
        if self._featurizer is not None:
            # print('Featurizer found: ', self._featurizer)
            model_state['featurizer'] = self._featurizer
        
        try:
            model_state.update(self.get_state_dict())
        except NotImplementedError:
            warnings.warn('get_state_dict() is not implemented. Using default implementation')

        return model_state

    def load_state_dict(self, state_dict:dict, *args, **kwargs): pass

    def infer_predict(self, logits: Union[object, torch.Tensor]): return logits

    def is_pytorch_module(self) -> bool: return self._model_class is not None and issubclass(self._model_class, nn.Module) and self._onnx is None

    def quantize(self):
        if self.is_pytorch_module():
            if self.model is None:
                self.init_model()
            quantize_model(self.model)

    def dequantize(self):
        if self.is_pytorch_module():
            if self.model is None:
                self.init_model()
            dequantize_model(self.model)

    def export_onnx(self, dummy_input, path='', print_graph=True, preserve_state=False):
        if self.is_pytorch_module():
            if self.model is None:
                self.init_model()
            torch.onnx.export(self.model, dummy_input, path, verbose=print_graph)
            self._onnx = path
            if not preserve_state:
                self.model = None
                self.config['state_dict'] = None
                self.save(path + '.bin')
            # if print_graph:
            #     import onnx
            #     onnx_model = onnx.load(path)
            #     onnx.helper.printable_graph(onnx_model.graph)

    def transform(self, X, interpret_fn:Callable = None, return_logits:bool = False, *args, **kwargs):
        if self._model is None and self._onnx_model is None: return
        is_pytorch = self.is_pytorch_module()

        rep = None
        if is_pytorch:
            self._model.eval()
        elif self._onnx_model is not None:
            import caffe2.python.onnx.backend as backend
            rep = backend.prepare(self._onnx_model, device='CPU')

        with torch.no_grad():
            if not torch.is_tensor(X) and self._featurizer is not None:
                X = self._featurizer.transform(X)
            
            X = self.preprocess_input(X)

            if X is None:
                logits = None
            else:
                if self._predict_fn is None:
                    if self._onnx_model is None:
                        logits = self._model(to_gpu(X))
                    else:
                        if torch.is_tensor(X):
                            X = X.numpy()

                        warnings.warn('Running inference as onnx model')
                        rep.run(X)
                        logits = torch.from_numpy(logits)
                else:
                    logits = self._predict_fn(X)
        
        if is_pytorch:
            self._model.train()
        
        if isinstance(logits, dict) and 'logits' in logits:
            logits = logits['logits']
        
        if return_logits or logits is None:
            return logits
        elif interpret_fn is not None:
            return interpret_fn(logits, *args, **kwargs)
        else:
            return self.infer_predict(logits, *args, **kwargs)
    
    def freeze_to(self, n:int):
        c = self.get_layer_groups()
        for l in c:
            set_trainable(l, False)
        for l in c[n:]:
            set_trainable(l, True)

    def freeze_all_but(self, n:int):
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

    @property
    def model(self): return self._model.module if (self._use_data_parallel == True and self._model is not None and hasattr(self._model, 'module')) else self._model

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

    def summary(self) -> str:
        return self.__str__()

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def __str__(self) -> str:
        return self._model.__str__() if self._model is not None else '<non-pytorch model>'


class ILearner(object):

    def __init__(self, 
        model_wrapper:IModel,
        optimizer_fn:Union[str, Callable]='adam', 
        optimizer_kwargs:dict={},
        auto_optimize:bool=True,
        preprocess_batch:bool=False,
        uneven_batch_size:bool=False,
        collate_fn:Callable=None):
        """
        data: Dataset or tuple (X_train, y_train)
        """
        self._data = None
        self._val_data = None
        self._model_wrapper = model_wrapper
        self._optimizer_kwargs = optimizer_kwargs
        self._verbose = 1
        self._optimizer_fn = None
        self._optimizer = None
        self._metrics = None
        self._current_epoch = 0
        self._batch_idx = 0
        self._batch_size = 0
        self._auto_optimize = auto_optimize
        self._preprocess_batch = preprocess_batch
        self._uneven_batch_size = uneven_batch_size
        self._collate_fn = collate_fn
        self._halt = False # prematurely halt training
        self.clip_grad = 0
        self.fp16 = False
        self.gradient_accumulation_steps = 1

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

    def on_model_before_init(self): pass
    """
    Triggered after model is initialized
    """
    def on_model_init(self): pass
    
    def on_training_end(self): pass

    def on_epoch_start(self): pass

    def on_epoch_end(self): pass
    
    # Runs the forward pass
    # Returns: (loss, logits) loss and raw results of the model
    
    def on_epoch(self, 
        X, y,
        gradient_accumulation_steps: int = 1) -> dict: return dict()

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
        training_data: Iterable = None,
        validation_data: Iterable = None,
        epochs: int = 1,
        minibatches: int = None,
        epoch_start: int = 0, 
        batch_size: int = 64, 
        shuffle: bool = True,
        optimize_on_cpu: bool = False,
        fp16: bool = False,
        gradient_accumulation_steps: int = 1,
        callbacks: Iterable[object] = [],
        clip_grad:float=0):

        if self._uneven_batch_size: batch_size = 1

        self._batch_size = batch_size

        self.clip_grad = clip_grad

        if gradient_accumulation_steps and 'gradient_accumulation_steps' in dict(inspect.getmembers(self.on_epoch.__func__.__code__))['co_varnames']:
            print('Gradient accumulation is supported by this class')
            self.gradient_accumulation_steps = gradient_accumulation_steps
        else:
            self.gradient_accumulation_steps = 1

        if training_data is not None: self.set_training_data(training_data)
        if validation_data is not None: self.set_validation_data(validation_data)

        for callback in callbacks: callback.set_learner(self)

        self._callbacks = callbacks or []
        self._n_epochs = epochs
        self._optimize_on_cpu = optimize_on_cpu

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

            if self._val_data is not None:
                X_test, y_test = self._val_data

                X_test = self.model_wrapper.preprocess_dataset_X(X_test)
                y_test = self.model_wrapper.preprocess_dataset_y(y_test)

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
                    
                    if self._val_data is not None:
                        test_dataset = BatchPreprocessedDataset(X_test, y_test,
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
                    
                    if self._val_data is not None:
                        test_dataset = BatchPreprocessedDataset(X_test, y_test,
                            input_process_fn=lambda _X: self.model_wrapper.preprocess_input(
                                self.model_wrapper._featurizer.transform(_X)
                            ),
                            output_process_fn=self.model_wrapper.preprocess_output,
                            batch_size=batch_size)
            else:
                if self.model_wrapper._featurizer is not None:
                    X = self.model_wrapper._featurizer.transform(X)

                    if self._val_data is not None:
                        X_test = self.model_wrapper._featurizer.transform(X_test)

                X = self.model_wrapper.preprocess_input(X)
                y = self.model_wrapper.preprocess_output(y)

                if self._val_data is not None:
                    X_test = self.model_wrapper.preprocess_input(X_test)

                if not self._uneven_batch_size:
                    dataset = GenericDataset(X, y)

                    if self._val_data is not None:
                        test_dataset = GenericDataset(X, y)
        else:
            dataset = self._data

            self.init_on_dataset(dataset)

            test_dataset = self._val_data

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
            try:
                mp.set_start_method('spawn')
            except:
                warnings.warn('Error orcurred in multiprocessing.set_start_method')

        if not self._uneven_batch_size:
            loader_kwargs = {
                'batch_size': batch_size, 
                'num_workers': cpu_count,
                'shuffle': shuffle
            }
            if USE_GPU:
                loader_kwargs['pin_memory'] = True
            if self._collate_fn is not None:
                loader_kwargs['collate_fn'] = self._collate_fn
            
            data_loader = DataLoader(dataset, **loader_kwargs)
            
            if self._val_data is not None:
                test_data_loader = DataLoader(test_dataset, **loader_kwargs)
        else:
            data_loader = [([X[idx]], [y[idx]]) for idx in range(len(X))]

            if self._val_data is not None:
                test_data_loader = [([X_test[idx]], [y_test[idx]]) for idx in range(len(X_test))]

        if self.model_wrapper._featurizer is not None:
            self.model_wrapper.config['input_shape'] = self.model_wrapper._featurizer.get_output_shape()

        self.on_model_before_init()
        if self.model_wrapper.model is None:
            self.model_wrapper.init_model()
        self.on_model_init()

        model = self.model_wrapper._model

        # optimizer must be initialized after the model
        if self.optimizer is None and self._auto_optimize:
            optim_params = [
                (n, param) for n, param in model.named_parameters()
                if param.requires_grad
            ]

            if self._optimize_on_cpu:
                optim_params = [
                    (n, param.clone().detach().to('cpu').requires_grad_()) \
                    for n, param in optim_params
                ] 

            self.optimizer = self._optimizer_fn(
                [p for n, p in optim_params],
                **self._optimizer_kwargs
            )

        if self.model_wrapper.is_pytorch_module() and not hasattr(self, 'criterion'):
            raise ValueError('Criterion must be set for the Learner class before training')

        # fp16
        if fp16:
            try:
                from apex import amp, optimizers
                from apex.multi_tensor_apply import multi_tensor_applier

                model, self.optimizer = amp.initialize(model, self.optimizer, 
                    opt_level="O1",
                    loss_scale="dynamic"
                )
                self.model_wrapper._model = model
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

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

                    if model is not None and self.model_wrapper.is_pytorch_module(): model.train()

                    args = to_gpu(X_batch), to_gpu(y_batch)
                    kwargs = {}
                    if gradient_accumulation_steps > 1:
                        kwargs['gradient_accumulation_steps'] = self.gradient_accumulation_steps

                    epoch_ret = self.on_epoch(*args, **kwargs)

                    if epoch_ret is not None:
                        if 'logits' in epoch_ret:
                            with torch.no_grad():
                                batch_metrics = self.calculate_metrics(epoch_ret['logits'], y_batch) or {}
                        else:
                            batch_metrics = {}

                        if 'loss' in epoch_ret:
                            epoch_loss = epoch_ret['loss']

                            # backward
                            if fp16:
                                with amp.scale_loss(epoch_loss, self.optimizer) as scaled_loss:
                                    scaled_loss.backward()

                                if self.clip_grad > 0:
                                    torch.nn.utils.clip_grad_norm_(
                                        amp.master_params(self.optimizer),
                                        self.clip_grad
                                    )
                            else:
                                epoch_loss.backward()

                                if self.clip_grad > 0:
                                    torch.nn.utils.clip_grad_norm_(
                                        model.parameters(), 
                                        self.clip_grad
                                    )

                            epoch_ret['loss'] = epoch_loss.detach().cpu().item()
                            batch_metrics['loss'] = epoch_ret['loss']
                        
                        self._batch_metrics = batch_metrics

                        if self._metrics is None:
                            self._metrics = batch_metrics
                        else:
                            self._metrics = {k: v + batch_metrics[k] for k, v in self._metrics.items()}

                    if self.model_wrapper.is_pytorch_module() and self._auto_optimize: 
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            if self._optimize_on_cpu:
                                is_nan = set_optimizer_params_grad(optim_params, model.named_parameters(), test_nan=True)

                                self.optimizer.step()
                                copy_optimizer_params_to_model(model.named_parameters(), optim_params)

                            else:
                                self.optimizer.step()

                            model.zero_grad()
                    
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