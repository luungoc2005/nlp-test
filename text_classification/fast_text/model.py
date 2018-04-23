import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.sparse
from torch.autograd import Variable
import numpy as np
from glove_utils import get_emb_matrix
from config import MAX_NUM_WORDS, NGRAM_BINS, EMBEDDING_DIM, SENTENCE_DIM
from fasttext_utils import _process_sentences

def process_sentences(sentences):
    words, ngrams = _process_sentences(list(sentences))
    return Variable(torch.from_numpy(words).long(), requires_grad=False), \
           Variable(torch.from_numpy(ngrams).long(), requires_grad=False)

class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.f = f

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

class FastText(nn.Module):

    def __init__(self,
                 hidden_size=100,
                 dropout_keep_prob=0.5,
                 classes=10):
        super(FastText, self).__init__()

        self.hidden_size = hidden_size
        self.dropout_keep_prob = dropout_keep_prob
        self.classes = classes

        # self.mean_embs = nn.EmbeddingBag(MAX_NUM_WORDS + 1, EMBEDDING_DIM, mode='mean', padding_idx=-1)
        emb_matrix = torch.from_numpy(get_emb_matrix()).float()
        self.word_embs = nn.Embedding.from_pretrained(emb_matrix)
        self.word_embs.padding_idx = 0
        self.word_embs.weight.requires_grad = False

        self.ngrams_embs = nn.Embedding(NGRAM_BINS, EMBEDDING_DIM, 
                                        padding_idx=0, sparse=True)
        self.ngrams_embs.weight.requires_grad = False
        self.highway = Highway(EMBEDDING_DIM * 2, 2, F.relu)

        self.i2o = nn.Linear(EMBEDDING_DIM * 2, self.classes)

        # self.i2h = nn.Linear(EMBEDDING_DIM * 2, self.hidden_size)
        # self.w2h = nn.Linear(emb_matrix.size(1), self.hidden_size // 2)
        # self.n2h = nn.Linear(EMBEDDING_DIM, self.hidden_size // 2)

        # self.h2o = nn.Linear(self.hidden_size, self.classes)

        self.init_weights()

    def init_weights(self):
        # nn.init.xavier_normal_(self.w2h.weight, gain=2)
        # nn.init.xavier_normal_(self.n2h.weight, gain=2)
        # nn.init.xavier_normal_(self.i2h.weight, gain=2)
        # nn.init.xavier_normal_(self.h2o.weight)
        nn.init.xavier_normal_(self.i2o.weight, gain=2)
        # self.i2h.bias.data.fill_(0)
        # self.w2h.bias.data.fill_(0)
        # self.n2h.bias.data.fill_(0)
        # self.h2o.bias.data.fill_(0)

    def forward(self, sequence, ngrams):
        embs = torch.mean(self.word_embs(sequence), dim=1)
        embs = F.dropout(embs, 1 - self.dropout_keep_prob)
        # embs = F.relu(self.w2h(embs))

        ngram_embs = torch.mean(self.ngrams_embs(ngrams), dim=1)
        ngram_embs = F.dropout(ngram_embs, 1 - self.dropout_keep_prob)
        # ngram_embs = F.relu(self.n2h(ngram_embs))

        x = torch.cat([embs, ngram_embs], dim=1)
        x = self.highway(x)
        # x = F.relu(self.i2h(x))
        # x = self.h2o(x)
        x = self.i2o(x)
        return x