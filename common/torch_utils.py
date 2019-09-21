import torch
import torch.nn as nn
import os
import warnings
import math

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    # return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def swish(x):
    return x * torch.sigmoid(x)

def cauchy(x):
    return 1 / (1 + x ** 2)

ACT2FN = {
    "gelu": gelu, 
    "relu": torch.nn.functional.relu,
    "tanh": torch.tanh,
    "cauchy": cauchy,
    "swish": swish
}

def children(m): return m if isinstance(m, (list, tuple)) else list(m.children())


def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module):
        f(m)
    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)


def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m, b))


def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b


def lr_schedule_slanted_triangular(step, n_epochs, max_lr=0.01, cut_frac=0.1, ratio=32):
    # https://arxiv.org/pdf/1801.06146.pdf
    cut = cut_frac * n_epochs
    if step < cut:
        p = step / cut
    else:
        p = 1 - (step - cut) / (cut * (1 / cut_frac - 1))
    return max_lr * (1 + p * (ratio - 1)) / ratio

USE_GPU = os.environ.get('USE_GPU', '').lower()
if USE_GPU == '':
    USE_GPU = torch.cuda.is_available()
else:
    # USE_GPU = (USE_GPU == 'True')
    pass

def use_data_parallel():
    return False # temporarily disable this
    # result = USE_GPU != '' and torch.cuda.device_count() > 1
    # if result:
    #     print('Using data parallel, number of GPUs: %s, devices: "%s"' % (str(torch.cuda.device_count()), str(USE_GPU)))
    # return result

def cuda_seed(seed):
    if USE_GPU: torch.cuda.manual_seed_all(seed)

device = None
def to_gpu(x, *args, **kwargs):
    global device
    '''puts pytorch variable to gpu, if cuda is available and USE_GPU is set to true. '''
    if device is None:
        if USE_GPU != '':
            if USE_GPU == True or USE_GPU == 'true':
                device = "cuda:0"
            elif USE_GPU == False or USE_GPU == 'false':
                device = "cpu"
            else:
                device = USE_GPU
        else:
            device = "cpu"
        device = torch.device(device)
    
    if hasattr(x, 'to'):
        return x.to(device, *args, **kwargs)
    else:
        return x

def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            warnings.warn("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            warnings.warn("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def make_variable_batch_size(num_inputs, onnx_model):
    """
    Changes the input batch dimension to a string, which makes it variable.
    Tensorflow interpretes this as the "?" shape.
    `num_inputs` must be specified because `onnx_model.graph.input` is a list
    of inputs of all layers and not just model inputs.
    :param num_inputs: int, Number of model inputs (e.g. 2 for Text and Image)
    :param onnx_model: ONNX model instance
    :return: ONNX model instance with variable input batch size
    """
    for i in range(num_inputs):
        onnx_model.graph.input[i].type.tensor_type.\
                                shape.dim[0].dim_param = 'batch_size'
    return onnx_model