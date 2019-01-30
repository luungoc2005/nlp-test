import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
from config import EMBEDDING_DIM, UNK_TAG
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from common.torch_utils import to_gpu
from common.utils import letterToIndex, n_letters, prepare_vec_sequence, word_to_vec, argmax

class BRNNWordEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int = None,
        letters_dim: int = None,
        dropout_keep_prob: float = 0.5,
        rnn_type: str ='GRU'
    ) -> None:
        super(BRNNWordEncoder, self).__init__()

        assert rnn_type in ['GRU', 'LSTM']

        self.hidden_dim = hidden_dim or EMBEDDING_DIM
        self.letters_dim = letters_dim or n_letters
        self.dropout_keep_prob = dropout_keep_prob

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

        words_batch = to_gpu(words_batch)

        words_batch = self.dropout(self.embedding(words_batch))

        # print('words_batch: %s' % str(words_batch.size()))
        # Sort by length (keep idx)
        word_lengths, idx_sort = np.sort(word_lengths)[::-1], np.argsort(-word_lengths)
        idx_unsort = np.argsort(idx_sort)

        idx_sort = to_gpu(torch.from_numpy(idx_sort))

        words_batch = words_batch.index_select(1, idx_sort)

        # Handling padding in Recurrent Networks
        # copy() call is to fix negative strides support in pytorch
        words_packed = pack_padded_sequence(words_batch, word_lengths.copy())
        words_output = self.rnn(words_packed)[0]
        words_output = pad_packed_sequence(words_output)[0]

        # Un-sort by length
        idx_unsort = to_gpu(torch.from_numpy(idx_unsort))

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

class CNNWordEncoder(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        kernel_sizes: List[int],
        hidden_dim: Optional[int] = None,
        letters_dim: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        super(CNNWordEncoder, self).__init__()

        self.hidden_dim = hidden_dim or EMBEDDING_DIM
        self.letters_dim = letters_dim or n_letters
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        self.embedding = nn.Embedding(n_letters + 1, self.hidden_dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.hidden_dim, 
                    self.out_channels, 
                    K, padding=K // 2
                )
                for K in self.kernel_sizes
            ]
        )
        self.output_dim = self.out_channels * len(self.kernel_sizes)

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        batch_size, max_sent_length, max_word_length = tuple(chars.size())
        chars = chars.view(batch_size * max_sent_length, max_word_length)

        # char_embedding: (bsize * max_sent_length, max_word_length, emb_size)
        char_embedding = self.char_embed(chars)

        # conv_inp dim: (bsize * max_sent_length, emb_size, max_word_length)
        conv_inp = char_embedding.transpose(1, 2)
        char_conv_outs = [F.relu(conv(conv_inp)) for conv in self.convs]

        # Apply max pooling
        # char_pool_out[i] dims: (bsize * max_sent_length, out_channels)
        char_pool_outs = [torch.max(out, dim=2)[0] for out in char_conv_outs]

        # Concat different feature maps together
        # char_pool_out dim: (bsize * max_sent_length, out_channel * num_kernels)
        char_pool_out = torch.cat(char_pool_outs, 1)

        # Reshape to (bsize, max_sent_length, out_channel * len(self.convs))
        return char_pool_out.view(batch_size, max_sent_length, -1)

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
class WeightDrop(nn.Module):
    def __init__(self, 
        module: nn.Module, 
        weights: list, 
        dropout: float = 0, 
        variational: bool = False):

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
        if issubclass(type(self.module), nn.RNNBase):
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

# lifted from pytext
class CRF(nn.Module):
    """
    Compute the log-likelihood of the input assuming a conditional random field
    model.
    Args:
        num_tags: The number of tags
    """

    def __init__(self, num_tags: int, ignore_index: Optional[int] = 0) -> None:
        if num_tags <= 0:
            raise ValueError(f"Invalid number of tags: {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        # Add two states at the end to accommodate start and end states
        # (i,j) element represents the probability of transitioning from state i to j
        self.transitions = nn.Parameter(torch.Tensor(num_tags + 2, num_tags + 2))
        self.start_tag = num_tags
        self.end_tag = num_tags + 1
        self.reset_parameters()
        # TODO Remove hardcoding to read from metadata
        self.ignore_index = ignore_index

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        self.transitions.data[:, self.start_tag] = -10000
        self.transitions.data[self.end_tag, :] = -10000

    def get_transitions(self):
        return self.transitions.data

    def set_transitions(self, transitions: torch.Tensor = None):
        self.transitions.data = transitions

    def forward(
        self,
        emissions: torch.FloatTensor,
        tags: torch.LongTensor,
        ignore_index: Optional[int] = 0,
        reduce: bool = True,
    ) -> torch.Tensor:
        """
        Compute log-likelihood of input.
        Args:
            emissions: Emission values for different tags for each input. The
                expected shape is batch_size * seq_len * num_labels. Padding is
                should be on the right side of the input.
            tags: Actual tags for each token in the input. Expected shape is
                batch_size * seq_len
        """
        self.ignore_index = ignore_index
        mask = self._make_mask_from_targets(tags)

        numerator = self._compute_joint_llh(emissions, tags, mask)
        denominator = self._compute_log_partition_function(emissions, mask)

        llh = numerator - denominator
        return llh if not reduce else torch.mean(llh)

    def decode(
        self, emissions: torch.FloatTensor, seq_lens: torch.LongTensor
    ) -> torch.Tensor:
        """
        Given a set of emission probabilities, return the predicted tags.
        Args:
            emissions: Emission probabilities with expected shape of
                batch_size * seq_len * num_labels
            seq_lens: Length of each input.
        """
        mask = self._make_mask_from_seq_lens(seq_lens)
        result = self._viterbi_decode(emissions, mask)
        return result

    def _compute_joint_llh(
        self,
        emissions: torch.FloatTensor,
        tags: torch.LongTensor,
        mask: torch.FloatTensor,
    ) -> torch.Tensor:
        seq_len = emissions.shape[1]

        # Log-likelihood for a given input is calculated by using the known
        # correct tag for each timestep and its respective emission value.
        # Since actual tags for each time step is also known, sum of transition
        # probabilities is also calculated.
        # Sum of emission and transition probabilities gives the final score for
        # the input.
        llh = self.transitions[self.start_tag, tags[:, 0]].unsqueeze(1)
        llh += emissions[:, 0, :].gather(1, tags[:, 0].view(-1, 1)) * mask[
            :, 0
        ].unsqueeze(1)

        for idx in range(1, seq_len):
            old_state, new_state = (
                tags[:, idx - 1].view(-1, 1),
                tags[:, idx].view(-1, 1),
            )
            emission_scores = emissions[:, idx, :].gather(1, new_state)
            transition_scores = self.transitions[old_state, new_state]
            llh += (emission_scores + transition_scores) * mask[:, idx].unsqueeze(1)

        # Index of the last tag is calculated by taking the sum of mask matrix
        # for each input row and subtracting 1 from the sum.
        last_tag_indices = mask.sum(1, dtype=torch.long) - 1
        last_tags = tags.gather(1, last_tag_indices.view(-1, 1))

        llh += self.transitions[last_tags.squeeze(1), self.end_tag].unsqueeze(1)

        return llh.squeeze(1)

    def _compute_log_partition_function(
        self, emissions: torch.FloatTensor, mask: torch.FloatTensor
    ) -> torch.Tensor:
        seq_len = emissions.shape[1]

        log_prob = emissions[:, 0].clone()
        log_prob += self.transitions[self.start_tag, : self.start_tag].unsqueeze(0)

        for idx in range(1, seq_len):
            broadcast_emissions = emissions[:, idx].unsqueeze(1)
            broadcast_transitions = self.transitions[
                : self.start_tag, : self.start_tag
            ].unsqueeze(0)
            broadcast_logprob = log_prob.unsqueeze(2)
            score = broadcast_logprob + broadcast_emissions + broadcast_transitions

            score = torch.logsumexp(score, 1)
            log_prob = score * mask[:, idx].unsqueeze(1) + log_prob.squeeze(1) * (
                1.0 - mask[:, idx].unsqueeze(1)
            )

        log_prob += self.transitions[: self.start_tag, self.end_tag].unsqueeze(0)
        return torch.logsumexp(log_prob.squeeze(1), 1)

    def _viterbi_decode(
        self, emissions: torch.FloatTensor, mask: torch.FloatTensor
    ) -> torch.Tensor:
        seq_len = emissions.shape[1]
        mask = mask.to(torch.uint8)

        log_prob = emissions[:, 0].clone()
        log_prob += self.transitions[self.start_tag, : self.start_tag].unsqueeze(0)

        # At each step, we need to keep track of the total score, as if this step
        # was the last valid step.
        end_scores = log_prob + self.transitions[
            : self.start_tag, self.end_tag
        ].unsqueeze(0)

        best_scores_list = []
        # If the element has only token, empty tensor in best_paths helps
        # torch.cat() from crashing
        best_paths_list = [to_gpu(torch.Tensor().long())]
        best_scores_list.append(end_scores.unsqueeze(1))

        for idx in range(1, seq_len):
            broadcast_emissions = emissions[:, idx].unsqueeze(1)
            broadcast_transmissions = self.transitions[
                : self.start_tag, : self.start_tag
            ].unsqueeze(0)
            broadcast_log_prob = log_prob.unsqueeze(2)

            score = broadcast_emissions + broadcast_transmissions + broadcast_log_prob

            max_scores, max_score_indices = torch.max(score, 1)

            best_paths_list.append(max_score_indices.unsqueeze(1))

            # Storing the scores incase this was the last step.
            end_scores = max_scores + self.transitions[
                : self.start_tag, self.end_tag
            ].unsqueeze(0)

            best_scores_list.append(end_scores.unsqueeze(1))
            log_prob = max_scores

        best_scores = torch.cat(best_scores_list, 1).float()
        best_paths = torch.cat(best_paths_list, 1)

        _, max_indices_from_scores = torch.max(best_scores, 2)

        valid_index_tensor = to_gpu(torch.tensor(0)).long()
        padding_tensor = to_gpu(torch.tensor(self.ignore_index)).long()

        # Label for the last position is always based on the index with max score
        # For illegal timesteps, we set as ignore_index
        labels = max_indices_from_scores[:, seq_len - 1]
        labels = self._mask_tensor(labels, 1.0 - mask[:, seq_len - 1], padding_tensor)

        all_labels = labels.unsqueeze(1).long()

        # For Viterbi decoding, we start at the last position and go towards first
        for idx in range(seq_len - 2, -1, -1):
            # There are two ways to obtain labels for tokens at a particular position.

            # Option 1: Use the labels obtained from the previous position to index
            # the path in present position. This is used for all positions except
            # last position in the sequence.
            # Option 2: Find the indices with maximum scores obtained during
            # viterbi decoding. This is used for the token at the last position

            # For option 1 need to convert invalid indices to 0 so that lookups
            # dont fail.
            indices_for_lookup = all_labels[:, -1].clone()
            indices_for_lookup = self._mask_tensor(
                indices_for_lookup,
                indices_for_lookup == self.ignore_index,
                valid_index_tensor,
            )

            # Option 1 is used here when previous timestep (idx+1) was valid.
            indices_from_prev_pos = (
                best_paths[:, idx, :]
                .gather(1, indices_for_lookup.view(-1, 1).long())
                .squeeze(1)
            )
            indices_from_prev_pos = self._mask_tensor(
                indices_from_prev_pos, (1.0 - mask[:, idx + 1]), padding_tensor
            )

            # Option 2 is used when last timestep was not valid which means idx+1
            # is the last position in the sequence.
            indices_from_max_scores = max_indices_from_scores[:, idx]
            indices_from_max_scores = self._mask_tensor(
                indices_from_max_scores, mask[:, idx + 1], padding_tensor
            )

            # We need to combine results from 1 and 2 as rows in a batch can have
            # sequences of varying lengths
            labels = torch.where(
                indices_from_max_scores == self.ignore_index,
                indices_from_prev_pos,
                indices_from_max_scores,
            )

            # Set to ignore_index if present state is not valid.
            labels = self._mask_tensor(labels, (1 - mask[:, idx]), padding_tensor)
            all_labels = torch.cat((all_labels, labels.view(-1, 1).long()), 1)

        return torch.flip(all_labels, [1])

    def _make_mask_from_targets(self, targets):
        mask = targets.ne(self.ignore_index).float()
        return mask

    def _make_mask_from_seq_lens(self, seq_lens):
        seq_lens = seq_lens.view(-1, 1)
        max_len = torch.max(seq_lens)
        range_tensor = to_gpu(torch.arange(max_len)).unsqueeze(0)
        range_tensor = range_tensor.expand(seq_lens.size(0), range_tensor.size(1))
        mask = (range_tensor < seq_lens).float()
        return mask

    def _mask_tensor(self, score_tensor, mask_condition, mask_value):
        masked_tensor = torch.where(mask_condition, mask_value, score_tensor)
        return masked_tensor