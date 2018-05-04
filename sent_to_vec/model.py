import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nltk.tokenize import word_tokenize
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from glove_utils import get_glove_data
from config import START_TAG, STOP_TAG, EMBEDDING_DIM, MAX_NUM_WORDS

def process_input(sentences):
    sentences = [
        [START_TAG] + word_tokenize(sent) + [STOP_TAG]
        for sent in sentences
    ]

    # Filter out words without word vectors
    glove_data = get_glove_data()
    
    for idx in range(len(sentences)):
        sent_filtered = [
            word for word in sentences[idx]
            if word in glove_data
        ]
        if len(sent_filtered) == 0:
            sent_filtered = [STOP_TAG]
        sentences[idx] = sent_filtered

    # Sort sentences by lengths
    lengths = np.array([len(sent) for sent in sentences])
    lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
    sentences = np.array(sentences)[idx_sort]

    return sentences, lengths, idx_sort

class MaskedConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 groups=1, bias=True, causal=True):
        if causal:
            padding = (kernel_size - 1) * dilation
        else:
            padding = (kernel_size - 1) * dilation // 2
        super(MaskedConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                           stride=1, padding=padding, dilation=dilation,
                                           groups=groups, bias=bias)

    def forward(self, inputs):
        output = super(MaskedConv1d, self).forward(inputs)
        return output[:, :, :inputs.size(2)]


class GatedConv1d(MaskedConv1d):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 groups=1, bias=True, causal=True):
        super(GatedConv1d, self).__init__(in_channels, 2 * out_channels,
                                          kernel_size, dilation, groups, bias, causal)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        output = super(GatedConv1d, self).forward(inputs)
        mask, output = output.chunk(2, 1)
        mask = self.sigmoid(mask)

        return output * mask

class StackedConv(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size=3,
                 num_layers=4, bias=True,
                 dropout=0, causal=True):
        super(StackedConv, self).__init__()
        self.convs = nn.ModuleList()
        size = input_size
        for l in range(num_layers):
            self.convs.append(GatedConv1d(size, hidden_size, 1, bias=bias,
                                          causal=False))
            self.convs.append(nn.BatchNorm1d(hidden_size))
            self.convs.append(MaskedConv1d(hidden_size, hidden_size,
                                           kernel_size, bias=bias,
                                           groups=hidden_size,
                                           causal=causal))
            self.convs.append(nn.BatchNorm1d(hidden_size))
            size = hidden_size

    def forward(self, x):
        res = None
        for conv in self.convs:
            x = conv(x)
            if res is not None:
                x = x + res
            res = x
        return x

"""
ConvNetEncoder
"""
class ConvNetEncoder(nn.Module):
    def __init__(self,
                 embedding_dim = None,
                 vocab_size = None,
                 hidden_dim = 4200,
                 is_cuda = None,
                 dropout_keep_prob = 1):
        super(ConvNetEncoder, self).__init__()

        self.embedding_dim = embedding_dim or EMBEDDING_DIM
        self.vocab_size = vocab_size or MAX_NUM_WORDS
        self.dropout_keep_prob = dropout_keep_prob
        self.hidden_dim = hidden_dim
        self.is_cuda = is_cuda or torch.cuda.is_available()

        self.dropout = nn.Dropout(1 - self.dropout_keep_prob)
        self.convs = StackedConv(self.embedding_dim, self.hidden_dim)

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: Variable(seqlen x batch x worddim)

        sent, sent_len = sent_tuple

        sent = sent.transpose(0,1).transpose(1,2).contiguous()
        sent = self.dropout(sent)

        sent = self.convs(sent)
        emb = torch.max(sent, 2)[0]

        return emb

class BiGRUEncoder(nn.Module):

    def __init__(self,
                 embedding_dim = None,
                 vocab_size = None,
                 hidden_dim = 4200,
                 is_cuda = None,
                 dropout_keep_prob = 1):
        super(BiGRUEncoder, self).__init__()

        self.embedding_dim = embedding_dim or EMBEDDING_DIM
        self.vocab_size = vocab_size or MAX_NUM_WORDS
        self.dropout_keep_prob = dropout_keep_prob
        self.hidden_dim = hidden_dim
        self.is_cuda = is_cuda or torch.cuda.is_available()

        self.lstm = nn.GRU(self.embedding_dim, self.hidden_dim, 1,
                                bidirectional=True, 
                                dropout=1-self.dropout_keep_prob)

    def forward(self, sent_tuple):
        sent, sent_len = sent_tuple

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort)

        if self.is_cuda:
            idx_sort = idx_sort.cuda()

        sent = sent.index_select(1, idx_sort)

        # Handling padding in Recurrent Networks
        sent_packed = pack_padded_sequence(sent, sent_len)
        sent_output = self.lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort)
        if self.is_cuda:
            idx_unsort = idx_unsort.cuda()
        sent_output = sent_output.index_select(1, idx_unsort)

        # Max Pooling
        embeds = torch.max(sent_output, 0)[0]
        if embeds.ndimension() == 3:
            embeds = embeds.squeeze(0)
            assert embeds.ndimension() == 2

        return embeds

class NLINet(nn.Module):
    
    def __init__(self,
                 lstm_dim = 4200,
                 hidden_dim = 512,
                 encoder = None):
        super(NLINet, self).__init__()

        self.lstm_dim = lstm_dim
        self.input_dim = 4 * 2 * lstm_dim
        self.hidden_dim = hidden_dim
        self.classes = 3

        self.encoder = encoder

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.classes)
        )

    def forward(self, sent1, sent2):
        # sent1/sent2 are tuples of (sentences, lengths)

        u = self.encoder(sent1)
        v = self.encoder(sent2)

        feats = torch.cat((u, v, torch.abs(u - v), u * v), 1)
        output = self.classifier(feats)

        return output

    def encode(self, sentence):
        return self.encoder(sentence)
