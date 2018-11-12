import torch
import numpy as np

def to_long_tensor(tensor):
    if tensor.ndimension() == 2 and \
        (tensor.dtype == torch.float32 or tensor.dtype == torch.float64):
        return torch.max(tensor, dim=-1)[1]
    else:
        return tensor.long()

def to_long_np(np_array):
    if np_array.dtype == np.float32 or np_array.dtype == np.float64:
        return np.argmax(np_array, -1)
    else:
        return np_array

def accuracy(preds, targs):
    targs = to_long_tensor(targs)
    preds = to_long_tensor(preds)
    return (preds==targs).float().mean()

def accuracy_np(preds, targs):
    targs = to_long_np(targs)
    preds = to_long_np(preds)
    return (preds==targs).mean()

def accuracy_thresh(thresh):
    return lambda preds,targs: accuracy_multi(preds, targs, thresh)

def accuracy_multi(preds, targs, thresh):
    return ((preds>thresh).float()==targs).float().mean()

def accuracy_multi_np(preds, targs, thresh):
    return ((preds>thresh)==targs).mean()

def recall(log_preds, targs, thresh=0.5, epsilon=1e-8):
    targs = to_long_tensor(targs)
    preds = torch.exp(log_preds)
    pred_pos = torch.max(preds > thresh, dim=1)[1]
    tpos = torch.mul((targs.byte() == pred_pos.byte()), targs.byte())
    return tpos.sum()/(targs.sum() + epsilon)

def recall_np(preds, targs, thresh=0.5, epsilon=1e-8):
    pred_pos = preds > thresh
    tpos = torch.mul((targs.byte() == pred_pos), targs.byte())
    return tpos.sum()/(targs.sum() + epsilon)

def precision(log_preds, targs, thresh=0.5, epsilon=1e-8):
    targs = to_long_tensor(targs)
    preds = torch.exp(log_preds)
    pred_pos = torch.max(preds > thresh, dim=1)[1]
    tpos = torch.mul((targs.byte() == pred_pos.byte()), targs.byte())
    return tpos.sum()/(pred_pos.sum() + epsilon)

def precision_np(preds, targs, thresh=0.5, epsilon=1e-8):
    pred_pos = preds > thresh
    tpos = torch.mul((targs.byte() == pred_pos), targs.byte())
    return tpos.sum()/(pred_pos.sum() + epsilon)

def fbeta(log_preds, targs, beta, thresh=0.5, epsilon=1e-8):
    """Calculates the F-beta score (the weighted harmonic mean of precision and recall).
    This is the micro averaged version where the true positives, false negatives and
    false positives are calculated globally (as opposed to on a per label basis).
    beta == 1 places equal weight on precision and recall, b < 1 emphasizes precision and
    beta > 1 favors recall.
    """
    assert beta > 0, 'beta needs to be greater than 0'
    beta2 = beta ** 2
    rec = recall(log_preds, targs, thresh)
    prec = precision(log_preds, targs, thresh)
    return (1 + beta2) * prec * rec / (beta2 * prec + rec + epsilon)

def fbeta_np(preds, targs, beta, thresh=0.5, epsilon=1e-8):
    """ see fbeta """
    assert beta > 0, 'beta needs to be greater than 0'
    beta2 = beta ** 2
    rec = recall_np(preds, targs, thresh)
    prec = precision_np(preds, targs, thresh)
    return (1 + beta2) * prec * rec / (beta2 * prec + rec + epsilon)

def f1(log_preds, targs, thresh=0.5): return fbeta(log_preds, targs, 1, thresh)
def f1_np(preds, targs, thresh=0.5): return fbeta_np(preds, targs, 1, thresh)