import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from config import EMBEDDING_DIM, UNK_TAG
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from common.utils import letterToIndex, n_letters, prepare_vec_sequence, word_to_vec, argmax

class BRNNWordEncoder(nn.Module):

    def __init__(self,
                 hidden_dim=None,
                 letters_dim=None,
                 dropout_keep_prob=0.5,
                 is_cuda=None,
                 rnn_type='GRU'):
        super(BRNNWordEncoder, self).__init__()

        assert rnn_type in ['GRU', 'LSTM']

        self.hidden_dim = hidden_dim or EMBEDDING_DIM
        self.letters_dim = letters_dim or n_letters
        self.dropout_keep_prob = dropout_keep_prob
        self.is_cuda = is_cuda if is_cuda is not None else torch.cuda.is_available()

        self.embedding = nn.Embedding(n_letters + 1, self.hidden_dim)
        self.dropout = nn.Dropout(1 - dropout_keep_prob)

        if rnn_type == 'GRU':
            self.rnn = nn.GRU(self.hidden_dim,
                self.hidden_dim // 2,
                num_layers=1,
                bidirectional=True)
        else:
            self.rnn = nn.LSTM(self.hidden_dim,
                self.hidden_dim // 2,
                num_layers=1,
                bidirectional=True)

    def forward(self, sentence):
        words_batch, word_lengths = self._process_sentence([
            token if len(token) > 0 else UNK_TAG
            for token in sentence
        ])

        if self.is_cuda:
            words_batch = words_batch.cuda()

        words_batch = self.dropout(self.embedding(words_batch))

        # print('words_batch: %s' % str(words_batch.size()))
        # Sort by length (keep idx)
        word_lengths, idx_sort = np.sort(word_lengths)[::-1], np.argsort(-word_lengths)
        idx_unsort = np.argsort(idx_sort)

        if self.is_cuda:
            idx_sort = torch.from_numpy(idx_sort).cuda()
        else:
            idx_sort = torch.from_numpy(idx_sort)

        words_batch = words_batch.index_select(1, idx_sort)

        # Handling padding in Recurrent Networks
        # copy() call is to fix negative strides support in pytorch
        words_packed = pack_padded_sequence(words_batch, word_lengths.copy())
        words_output = self.rnn(words_packed)[0]
        words_output = pad_packed_sequence(words_output)[0]

        # Un-sort by length
        if self.is_cuda:
            idx_unsort = torch.from_numpy(idx_unsort).cuda()
        else:
            idx_unsort = torch.from_numpy(idx_unsort)

        words_output = words_output.index_select(1, idx_unsort)

        # Max Pooling
        embeds = torch.max(words_output, 0)[0]
        if embeds.ndimension() == 3:
            embeds = embeds.squeeze(0)
            assert embeds.ndimension() == 2

        # print(embeds)

        return embeds

    def get_layer_groups(self):
        return [(self.embedding, self.dropout), self.rnn]
    
    def _letter_to_array(self, letter):
        ret_val = np.zeros(1, n_letters)
        ret_val[0][letterToIndex(letter)] = 1
        return ret_val


    def _word_to_array(self, word):
        ret_val = np.zeros(len(word), 1, n_letters)
        for li, letter in enumerate(word):
            ret_val[li][0][letterToIndex(letter)] = 1
        return ret_val

    def _process_sentence(self, sentence):
        word_lengths = np.array([len(word) for word in sentence])
        max_len = np.max(word_lengths)
        words_batch = np.zeros((max_len, len(sentence)))

        for i in range(len(sentence)):
            for li, letter in enumerate(sentence[i]):
                words_batch[li][i] = letterToIndex(letter)

        words_batch = torch.from_numpy(words_batch).long()
        return words_batch, word_lengths


# class ConvNetWordEncoder(nn.Module):

#     def __init__(self,
#                  hidden_dim=None,
#                  letters_dim=None,
#                  num_filters=None,
#                  dropout_keep_prob=0.5,
#                  is_cuda=None):
#         super(ConvNetWordEncoder, self).__init__()

#         # https://arxiv.org/pdf/1603.01354.pdf
#         self.hidden_dim = hidden_dim or EMBEDDING_DIM
#         self.letters_dim = letters_dim or n_letters
#         self.num_filters = num_filters or 30
#         self.dropout_keep_prob = dropout_keep_prob
#         self.embedding = nn.Embedding(n_letters, self.hidden_dim)
#         self.is_cuda = is_cuda if is_cuda is not None else torch.cuda.is_available()

#         self.convs = []
#         for _ in range(self.num_filters):
#             self.convs.append(
#                 nn.Sequential(
#                     nn.Conv1d(self.hidden_dim, self.hidden_dim // self.num_filters,
#                               kernel_size=3, stride=1, padding=1),
#                     nn.ReLU(inplace=True),
#                     nn.Dropout(1 - self.dropout_keep_prob)
#                 )
#             )

#     def forward(self, sentence):
#         words_batch, _ = _process_sentence(sentence)

#         if self.is_cuda:
#             words_batch = words_batch.cuda()

#         words_batch = self.embedding(words_batch)
#         words_batch = words_batch.transpose(0, 1).transpose(1, 2).contiguous()

#         convs_batch = []
#         for conv in self.convs:
#             conv_batch = conv(words_batch)
#             convs_batch.append(torch.max(conv_batch, 2)[0])

#         embeds = torch.cat(convs_batch, 1)

#         return embeds

#     def get_layer_groups(self):
#         return [(self.embedding, self.dropout), *zip(self.convs)]

class Highway(nn.Module):

    def __init__(self, size, num_layers, f):
        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.f = f

        self.init_weights()

    def init_weights(self):
        for layer in range(self.num_layers):
            nn.init.xavier_normal_(self.nonlinear[layer].weight)
            nn.init.xavier_normal_(self.linear[layer].weight, gain=2)
            nn.init.xavier_normal_(self.gate[layer].weight)

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        m.requires_grad = False
        mask = m / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

# https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py
class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            # print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.ones(raw_w.size(0), 1)
                if raw_w.is_cuda: mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
}


class BertAdam(optim.Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay_rate: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay_rate=0.01,
                 max_grad_norm=1.0):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay_rate=weight_decay_rate,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay_rate'] > 0.0:
                    update += group['weight_decay_rate'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss