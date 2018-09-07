import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import EMBEDDING_DIM, UNK_TAG
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from common.utils import letterToIndex, n_letters, prepare_vec_sequence, word_to_vec, argmax, log_sum_exp

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
        words_packed = pack_padded_sequence(words_batch, word_lengths)
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