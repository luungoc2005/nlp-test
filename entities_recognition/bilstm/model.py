import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import START_TAG, STOP_TAG, EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS

from common.utils import letterToIndex, n_letters, prepare_vec_sequence, word_to_vec, to_scalar, argmax, prepare_sequence, log_sum_exp

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
    words_batch = np.zeros((max_len, len(sentence), n_letters))

    for i in range(len(sentence)):
        for li, letter in enumerate(sentence[i]):
            words_batch[li][i][letterToIndex(letter)] = 1.
    
    words_batch = Variable(torch.from_numpy(words_batch).float())
    return words_batch, word_lengths

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

class BLSTMWordEncoder(nn.Module):

    def __init__(self, 
                 hidden_dim = None,
                 letters_dim = None):
        super(BLSTMWordEncoder, self).__init__()

        self.hidden_dim = hidden_dim or EMBEDDING_DIM
        self.letters_dim = letters_dim or n_letters

        self.rnn = nn.LSTM(n_letters,
                           self.hidden_dim // 2,
                           num_layers=1,
                           bidirectional=True)

    def forward(self, sentence):
        words_batch, word_lengths = _process_sentence(sentence)

        # Sort by length (keep idx)
        word_lengths, idx_sort = np.sort(word_lengths)[::-1], np.argsort(-word_lengths)
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort)

        words_batch = words_batch.index_select(1, Variable(idx_sort))

        # Handling padding in Recurrent Networks
        words_packed = pack_padded_sequence(words_batch, word_lengths)
        words_output = self.rnn(words_packed)[0]
        words_output = pad_packed_sequence(words_output)[0]
        
        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort)
        words_output = words_output.index_select(1, Variable(idx_unsort))

        # Max Pooling
        embeds = torch.max(words_output, 0)[0]
        if embeds.ndimension() == 3:
            embeds = embeds.squeeze(0)
            assert embeds.ndimension() == 2

        # print(embeds)

        return embeds

class ConvNetWordEncoder(nn.Module):

    def __init__(self,
                 hidden_dim = None,
                 letters_dim = None,
                 num_filters = None,
                 dropout_keep_prob = 0.5):
        super(ConvNetWordEncoder, self).__init__()

        # https://arxiv.org/pdf/1603.01354.pdf
        self.hidden_dim = hidden_dim or EMBEDDING_DIM
        self.letters_dim = letters_dim or n_letters
        self.num_filters = num_filters or 30
        self.dropout_keep_prob = dropout_keep_prob

        self.convs = []
        for _ in range(self.num_filters):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(self.letters_dim, self.hidden_dim // self.num_filters,
                            kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Dropout(1 - self.dropout_keep_prob)
                )
            )

    def forward(self, sentence):
        words_batch, _ = _process_sentence(sentence)
        
        words_batch = words_batch.transpose(0, 1).transpose(1, 2).contiguous()

        convs_batch = []
        for conv in self.convs:
            conv_batch = conv(words_batch)
            convs_batch.append(torch.max(conv_batch, 2)[0])

        embeds = torch.cat(convs_batch, 1)

        return embeds

class BiLSTM_CRF(nn.Module):

    def __init__(self, 
                 tag_to_ix,
                 embedding_dim = None,
                 char_embedding_dim = None,
                 hidden_dim = None,
                 num_layers = None,
                 dropout_keep_prob = 0.5):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim or EMBEDDING_DIM
        self.char_embedding_dim = char_embedding_dim or CHAR_EMBEDDING_DIM
        self.hidden_dim = hidden_dim or HIDDEN_DIM
        self.tag_to_ix = tag_to_ix
        self.num_layers = num_layers or NUM_LAYERS
        self.dropout_keep_prob = dropout_keep_prob
        self.tagset_size = len(tag_to_ix)

        self.word_encoder = BLSTMWordEncoder(self.char_embedding_dim)
        self.highway = Highway(self.embedding_dim + self.char_embedding_dim, 2, F.relu)
        self.dropout = nn.Dropout(1 - self.dropout_keep_prob)
        self.lstm = nn.LSTM(self.embedding_dim + self.char_embedding_dim,
                            self.hidden_dim // 2,
                            num_layers=self.num_layers,
                            bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.randn(self.num_layers * 2, 1, self.hidden_dim // 2)),
                Variable(torch.randn(self.num_layers * 2, 1, self.hidden_dim // 2)))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = Variable(init_alphas)

        # Iterate through the sentence
        for feat in feats:
            emit_score = feat.view(-1, 1)
            tag_var = forward_var + self.transitions + emit_score
            max_tag_var, _ = torch.max(tag_var, dim=1)
            tag_var = tag_var - max_tag_var.view(-1, 1)
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1)
        
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        seq_len = len(sentence)

        embeds = sentence.view(seq_len, 1, -1) #[seq_len, batch_size, features]
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(seq_len, self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        for i, feat in enumerate(feats):
            # print((len(feats), len(self.transitions), len(tags)))
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = Variable(init_vvars)
        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()
            next_tag_var = next_tag_var.data.cpu().numpy()
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            viterbivars_t = Variable(torch.FloatTensor(viterbivars_t))
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var.data[self.tag_to_ix[STOP_TAG]] = -10000.
        terminal_var.data[self.tag_to_ix[START_TAG]] = -10000.
        
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        path_score = terminal_var[best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        word_embeds = prepare_vec_sequence(sentence, word_to_vec, output='variable')
        char_embeds = self.word_encoder(sentence)
        sentence_in = torch.cat((word_embeds, char_embeds), dim=1)
        sentence_in = self.highway(sentence_in)
        sentence_in = self.dropout(sentence_in)

        feats = self._get_lstm_features(sentence_in)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        word_embeds = prepare_vec_sequence(sentence, word_to_vec, output='variable')
        char_embeds = self.word_encoder(sentence)
        sentence_in = torch.cat((word_embeds, char_embeds), dim=1)
        sentence_in = self.highway(sentence_in)

        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence_in)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq