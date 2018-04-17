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

class FastText(nn.Module):

    def __init__(self,
                 input_size=None,
                 hidden_size=100,
                 dropout_keep_prob=0.5,
                 classes=10):
        super(FastText, self).__init__()

        self.input_size = input_size or EMBEDDING_DIM * 2
        self.hidden_size = hidden_size
        self.dropout_keep_prob = dropout_keep_prob
        self.classes = classes

        # self.mean_embs = nn.EmbeddingBag(MAX_NUM_WORDS + 1, EMBEDDING_DIM, mode='mean', padding_idx=-1)
        emb_matrix = torch.from_numpy(get_emb_matrix()).float()
        self.word_embs = nn.Embedding.from_pretrained(emb_matrix)
        self.word_embs.padding_idx = 0

        self.ngrams_embs = nn.Embedding(NGRAM_BINS, EMBEDDING_DIM, padding_idx=0)
        self.ngrams_embs.requires_grad = False
        
        self.i2h = nn.Linear(self.input_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, self.classes)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.h2o.weight)
        self.h2o.bias.data.fill_(0)

    def forward(self, sequence, ngrams):
        embs = torch.mean(self.word_embs(sequence), dim=1)
        ngram_embs = torch.mean(self.ngrams_embs(ngrams), dim=1)
        x = torch.cat([embs, ngram_embs], dim=1)
        x = F.relu(self.i2h(x))
        x = F.dropout(x, 1 - self.dropout_keep_prob)
        x = self.h2o(x)
        return x