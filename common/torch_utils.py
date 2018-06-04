import torch.nn as nn


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
