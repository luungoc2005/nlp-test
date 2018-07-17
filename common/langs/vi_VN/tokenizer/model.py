import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import START_TAG, STOP_TAG, EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS
from common.langs.vi_VN.utils import remove_tone_marks
from common.torch_utils import set_trainable, children
from common.utils import letterToIndex, n_letters, prepare_vec_sequence, word_to_vec, argmax, log_sum_exp

def _letter_to_array(letter):
    ret_val = np.zeros(1, n_letters)
    ret_val[0][letterToIndex(letter)] = 1
    return ret_val


def _word_to_array(word):
    ret_val = np.zeros(len(word), 1, n_letters)
    for li, letter in enumerate(word):
        ret_val[li][0][letterToIndex(letter)] = 1
    return ret_val


def _process_sentence(sentence):
    word_lengths = np.array([len(word) for word in sentence])
    max_len = np.max(word_lengths)
    words_batch = np.zeros((max_len, len(sentence)))

    for i in range(len(sentence)):
        for li, letter in enumerate(sentence[i]):
            words_batch[li][i] = letterToIndex(letter)

    words_batch = Variable(torch.from_numpy(words_batch).long())
    return words_batch, word_lengths


class BGRUWordEncoder(nn.Module):

    def __init__(self,
                 hidden_dim=None,
                 letters_dim=None,
                 dropout_keep_prob=0.5,
                 is_cuda=None):
        super(BGRUWordEncoder, self).__init__()

        self.hidden_dim = hidden_dim or EMBEDDING_DIM
        self.letters_dim = letters_dim or n_letters
        self.dropout_keep_prob = dropout_keep_prob
        self.is_cuda = is_cuda or torch.cuda.is_available()

        self.embedding = nn.Embedding(n_letters + 1, self.hidden_dim)
        self.dropout = nn.Dropout(1 - dropout_keep_prob)
        self.rnn = nn.GRU(self.hidden_dim,
                          self.hidden_dim // 2,
                          num_layers=1,
                          bidirectional=True)

    def forward(self, sentence):
        words_batch, word_lengths = _process_sentence(sentence)

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

class BiLSTMTagger(nn.Module):

    def __init__(self,
                 max_emb_words=10000,
                 embedding_dim=100,
                 char_embedding_dim=200,
                 tokenizer=None,
                 hidden_dim=None,
                 num_layers=None,
                 dropout_keep_prob=0.8,
                 is_cuda=None):
        super(BiLSTMTagger, self).__init__()
        self.max_emb_words = max_emb_words
        self.embedding_dim = embedding_dim or EMBEDDING_DIM
        self.char_embedding_dim = char_embedding_dim or CHAR_EMBEDDING_DIM
        self.hidden_dim = hidden_dim or HIDDEN_DIM
        self.num_layers = num_layers or NUM_LAYERS
        self.dropout_keep_prob = dropout_keep_prob
        self.is_cuda = is_cuda or torch.cuda.is_available()

        self.word_encoder = BGRUWordEncoder(self.char_embedding_dim)

        if self.is_cuda:
            self.word_encoder = self.word_encoder.cuda()

        self.dropout = nn.Dropout(1 - self.dropout_keep_prob)
        self.embedding = nn.Embedding(self.max_emb_words + 1, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim + self.char_embedding_dim,
                            self.hidden_dim // 2,
                            num_layers=self.num_layers,
                            bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim, 1)
        self.hidden = self.init_hidden()

        # Set tokenizer
        self.tokenizer = tokenizer

    def init_hidden(self):
        hidden_0 = torch.randn(self.num_layers * 2, 1, self.hidden_dim // 2)
        hidden_1 = torch.randn(self.num_layers * 2, 1, self.hidden_dim // 2)

        if self.is_cuda:
            hidden_0 = hidden_0.cuda()
            hidden_1 = hidden_1.cuda()

        return hidden_0, hidden_1

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
        return [
            *zip(self.word_encoder.get_layer_groups()),
            self.lstm,
            self.hidden2tag
        ]

    def forward(self, sentence):
        tokens = self.tokenizer.texts_to_sequences([sentence])

        tokens = torch.LongTensor(tokens)
        # print('tokens: %s' % str(tokens.size()))
        if self.is_cuda:
            tokens = tokens.cuda()

        word_embeds = self.embedding(tokens).permute(1, 0, 2)
        # print('word_embeds: %s' % str(word_embeds.size()))

        char_embeds = self.word_encoder([
            remove_tone_marks(token) for token in sentence
        ]).unsqueeze(1)
        # print('char_embeds: %s' % str(char_embeds.size()))

        sentence_in = torch.cat((word_embeds, char_embeds), dim=-1)

        # Get the emission scores from the BiLSTM
        self.hidden = self.init_hidden()
        seq_len = len(sentence_in)

        # embeds = sentence_in.view(seq_len, 1, -1)  # [seq_len, batch_size, features]
        lstm_out, self.hidden = self.lstm(sentence_in, self.hidden)
        lstm_out = lstm_out.view(seq_len, self.hidden_dim)
        tags = self.hidden2tag(lstm_out).squeeze(1)

        return tags
