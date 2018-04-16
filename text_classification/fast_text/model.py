import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.sparse
from torch.autograd import Variable
import numpy as np
from hashlib import md5
from common.utils import wordpunct_tokenize, prepare_sequence, word_to_vec
from common.glove_utils import get_text_to_ix, get_emb_matrix
from config import GLOVE_PATH, MAX_NUM_WORDS, NGRAM_BINS, EMBEDDING_DIM, SENTENCE_DIM

# TODO: Rewrite this class to use nn.Embedding instead
# Just in case
def md5_hash_function(word, lower=True):
    hash_word = word.lower() if lower else word
    return int(md5(hash_word.encode()).hexdigest(), 16)

def bigram_hash(sequence, idx, buckets, lower=True):
    if idx - 1 >= 0:
        bigram = sequence[idx - 1] + sequence[idx]
        if lower: bigram = bigram.lower()
        return (hash(bigram) % (buckets - 1) + 1)
    else:
        return 0

def sentence_vector(sentence, maxlen=None):
    tokens = wordpunct_tokenize(sentence)
    words_sequence = prepare_sequence(tokens, get_text_to_ix())
    words_sequence.requires_grad = False
    ngrams_sequence = []

    # n-gram features
    # TODO: experiment with bigrams + trigrams - only bigrams for now
    for idx in range(len(tokens)):
        ngrams_sequence.append(bigram_hash(tokens, idx, NGRAM_BINS))
    
    ngrams_sequence = Variable(torch.LongTensor(ngrams_sequence), requires_grad=False)
    return (words_sequence, ngrams_sequence)

def process_sentences(sentences):
    words_seqs = []
    ngrams_seqs = []
    max_words = 0
    max_ngrams = 0
    for sent in sentences:
        words_seq, ngrams_seq = sentence_vector(sent)

        if words_seq.size(0) > max_words:
            max_words = words_seq.size(0)
        if ngrams_seq.size(0) > max_ngrams:
            max_ngrams = ngrams_seq.size(0)
        
        words_seqs.append(words_seq)
        ngrams_seqs.append(ngrams_seq)
    
    words_seqs = [
        F.pad(word_seq, (0, max_words - word_seq.size(0)), value=0).unsqueeze(0)
        for word_seq in words_seqs
    ]
    ngrams_seqs = [
        F.pad(ngram_seq, (0, max_words - ngram_seq.size(0)), value=0).unsqueeze(0)
        for ngram_seq in ngrams_seqs
    ]

    return torch.cat(words_seqs, dim=0), torch.cat(ngrams_seqs, dim=0)

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
        self.word_embs = nn.Embedding.from_pretrained(get_emb_matrix())
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