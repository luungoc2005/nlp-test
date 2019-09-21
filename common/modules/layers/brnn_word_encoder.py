import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Union
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

        words_batch = to_gpu(words_batch) # letters x words

        words_batch = self.dropout(self.embedding(words_batch)) # letters x words x embeds

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

        return embeds # words x embeds

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
