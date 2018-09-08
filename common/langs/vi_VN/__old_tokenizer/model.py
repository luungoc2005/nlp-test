import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, UNK_TAG
from common.langs.vi_VN.utils import remove_tone_marks
from common.torch_utils import set_trainable, children
from common.modules import BRNNWordEncoder

class BiLSTMTagger(nn.Module):

    def __init__(self,
                 max_emb_words=10000,
                 embedding_dim=100,
                 char_embedding_dim=200,
                 tokenizer=None,
                 hidden_dim=None,
                 num_layers=None,
                 dropout_keep_prob=0.6,
                 is_cuda=None):
        super(BiLSTMTagger, self).__init__()
        self.max_emb_words = max_emb_words
        self.embedding_dim = embedding_dim or EMBEDDING_DIM
        self.char_embedding_dim = char_embedding_dim or CHAR_EMBEDDING_DIM
        self.hidden_dim = hidden_dim or 600
        self.num_layers = num_layers or 3
        self.dropout_keep_prob = dropout_keep_prob
        self.is_cuda = is_cuda if is_cuda is not None else torch.cuda.is_available()

        self.word_encoder = BRNNWordEncoder(self.char_embedding_dim, rnn_type='GRU')

        if self.is_cuda:
            self.word_encoder = self.word_encoder.cuda()

        self.dropout = nn.Dropout(1 - self.dropout_keep_prob)

        # 0: reserved index by Keras tokenizer
        # num_words + 1: index for oov token
        self.embedding = nn.Embedding(self.max_emb_words + 2, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim + self.char_embedding_dim,
                            self.hidden_dim // 2,
                            num_layers=self.num_layers,
                            bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim, 1)

        # Set tokenizer
        self.tokenizer = tokenizer

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
        # print(tokens)
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

        seq_len = len(sentence_in)

        # embeds = sentence_in.view(seq_len, 1, -1)  # [seq_len, batch_size, features]
        lstm_out, _ = self.lstm(sentence_in)
        lstm_out = lstm_out.view(seq_len, self.hidden_dim)
        tags = self.hidden2tag(lstm_out).squeeze(1)

        return tags
