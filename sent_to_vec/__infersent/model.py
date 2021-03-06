import torch
import torch.nn as nn
import numpy as np
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from glove_utils import get_glove_data, get_word_vector
from common.word_vectors import get_word_vector
from config import START_TAG, STOP_TAG, EMBEDDING_DIM, MAX_NUM_WORDS
from common.torch_utils import set_trainable, children
from common.modules import Highway
from torchqrnn import QRNN

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


class ConvNetEncoder(nn.Module):
    """
    ConvNetEncoder
    """

    def __init__(self,
                 embedding_dim=None,
                 vocab_size=None,
                 hidden_dim=2400,
                 is_cuda=None,
                 dropout_keep_prob=1):
        super(ConvNetEncoder, self).__init__()

        self.embedding_dim = embedding_dim or EMBEDDING_DIM
        self.vocab_size = vocab_size or MAX_NUM_WORDS
        self.dropout_keep_prob = dropout_keep_prob
        self.hidden_dim = hidden_dim
        self.is_cuda = is_cuda if is_cuda is not None else torch.cuda.is_available()

        self.dropout = nn.Dropout(1 - self.dropout_keep_prob)
        self.convs = StackedConv(self.embedding_dim, self.hidden_dim)

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: Variable(seqlen x batch x worddim)

        sent, _ = sent_tuple

        sent = sent.transpose(0, 1).transpose(1, 2).contiguous()
        sent = self.dropout(sent)

        sent = self.convs(sent)
        emb = torch.max(sent, 2)[0]

        return emb


class BiLSTMEncoder(nn.Module):

    def __init__(self,
                 embedding_dim=None,
                 vocab_size=None,
                 hidden_dim=2400,
                 is_cuda=None,
                 dropout_keep_prob=1):
        super(BiLSTMEncoder, self).__init__()

        self.embedding_dim = embedding_dim or EMBEDDING_DIM
        self.vocab_size = vocab_size or MAX_NUM_WORDS
        self.dropout_keep_prob = dropout_keep_prob
        self.hidden_dim = hidden_dim
        self.is_cuda = is_cuda if is_cuda is not None else torch.cuda.is_available()

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, 1,
                            bidirectional=True,
                            dropout=1-self.dropout_keep_prob)

    def get_layer_groups(self):
        return [
            *zip(self.word_encoder.get_layer_groups()),
        ]

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


class QRNNEncoder(nn.Module):

    def __init__(self,
                 embedding_dim=None,
                 vocab_size=None,
                 hidden_dim=2400,
                 num_layers=3,
                 dropout_keep_prob=0.6,
                 pool_type='mean',
                 is_cuda=None):
        super(QRNNEncoder, self).__init__()

        assert pool_type in ['max', 'mean']

        self.pool_type = pool_type
        self.embedding_dim = embedding_dim or EMBEDDING_DIM
        self.vocab_size = vocab_size or MAX_NUM_WORDS
        self.dropout_keep_prob = dropout_keep_prob
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.is_cuda = is_cuda if is_cuda is not None else torch.cuda.is_available()

        self.qrnn = QRNN(self.embedding_dim, self.hidden_dim, self.num_layers,
                         dropout=1-self.dropout_keep_prob) # Outputs: output, h_n

    def get_layer_groups(self):
        return [
            *zip(self.word_encoder.get_layer_groups()),
        ]

    def forward(self, sent_tuple):
        sent, sent_len = sent_tuple

        sent_output = self.qrnn(sent)[1] #h_n: num_layers * num_directions, batch, hidden_size

        # Max/Mean Pooling
        if self.pool_type == 'max':
            sent_output = torch.max(sent_output, 0)[0]
            if sent_output.ndimension() == 3:
                sent_output = sent_output.squeeze(0)
                assert sent_output.ndimension() == 2
        else:
            # Might not work well due to no sequence masking (QRNN)
            sent_len = torch.FloatTensor(sent_len.copy()).unsqueeze(1)
            if self.is_cuda:
                sent_len = sent_len.cuda()
            sent_output = torch.sum(sent_output, 0)
            if sent_output.ndimension() == 3: # might not be necessary?
                sent_output = sent_output.squeeze(0)
                assert sent_output.ndimension() == 2
            sent_output = sent_output / sent_len.expand_as(sent_output)

        return sent_output


class QRNNEncoderConcat(nn.Module):

    def __init__(self,
                 embedding_dim=None,
                 vocab_size=None,
                 hidden_dim=2400,
                 num_layers=3,
                 is_cuda=None,
                 dropout_keep_prob=0.6):
        super(QRNNEncoderConcat, self).__init__()

        assert hidden_dim % num_layers == 0, 'Number of hidden dims must be divisable by number of layers'

        self.embedding_dim = embedding_dim or EMBEDDING_DIM
        self.vocab_size = vocab_size or MAX_NUM_WORDS
        self.dropout_keep_prob = dropout_keep_prob
        self.num_layers = num_layers
        self.hidden_dim = int(hidden_dim / self.num_layers)
        self.is_cuda = is_cuda if is_cuda is not None else torch.cuda.is_available()

        self.qrnn = QRNN(self.embedding_dim, self.hidden_dim, 
                         self.num_layers, dropout=1-self.dropout_keep_prob) # Outputs: output, h_n

    def get_layer_groups(self):
        return [
            *zip(self.word_encoder.get_layer_groups()),
        ]

    def forward(self, sent_tuple):
        sent, _ = sent_tuple

        sent_output = self.qrnn(sent)[1] #h_n: num_layers * num_directions, batch, hidden_size
        sent_output = sent_output.permute(1, 0, 2) # batch, layers, hidden
        sent_output = sent_output.reshape(sent_output.size(0), self.num_layers * self.hidden_dim)

        return sent_output


class NLINet(nn.Module):
    
    def __init__(self,
                 lstm_dim=2400,
                 hidden_dim=512,
                 bidirectional_encoder=True,
                 encoder=None):
        super(NLINet, self).__init__()

        self.lstm_dim = lstm_dim
        self.input_dim = 4 * 2 * lstm_dim if bidirectional_encoder else 4 * lstm_dim
        self.hidden_dim = hidden_dim
        self.classes = 3

        self.encoder = encoder

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.classes)
        )

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
            (self.highway, self.dropout),
            self.lstm,
            self.hidden2tag
        ]

    def forward(self, sent1, sent2):
        # sent1/sent2 are tuples of (sentences, lengths)

        u = self.encoder(sent1)
        v = self.encoder(sent2)

        feats = torch.cat((u, v, torch.abs(u - v), u * v), 1)
        output = self.classifier(feats)

        return output

    def encode(self, sentence):
        return self.encoder(sentence)
    
    def process_batch(self, batch):
        lengths = np.array([len(sent) for sent in batch])
        max_len = np.max(lengths)
        # embeds = np.zeros((max_len, len(batch), EMBEDDING_DIM))

        # for i in range(len(batch)):
        #     for j in range(len(batch[i])):
        #         vec = get_word_vector(batch[i][j])
        #         if vec is not None:
        #             embeds[j, i, :] = vec
        embeds = get_word_vector(batch)

        return torch.from_numpy(embeds).float().permute(1,0,2), lengths

    def process_input(self, sentences):
        # Filter out words without word vectors
        # glove_data = get_glove_data()
        
        sentences = [
            [START_TAG] + [word for word in word_tokenize(sent)] + [STOP_TAG]
            for sent in sentences
        ]
        sentences = [sent if len(sent) > 2 else [STOP_TAG] for sent in sentences]

        # Sort sentences by lengths
        # lengths = np.array([len(sent) for sent in sentences])
        # lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        # sentences = np.array(sentences)[idx_sort]

        return sentences #, lengths, idx_sort

