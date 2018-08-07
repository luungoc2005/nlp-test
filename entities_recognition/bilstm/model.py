import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import START_TAG, STOP_TAG, EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS
from common.torch_utils import set_trainable, children
from common.utils import prepare_vec_sequence, word_to_vec, argmax, log_sum_exp
from common.modules import BRNNWordEncoder

class BiLSTM_CRF(nn.Module):

    def __init__(self,
                 tag_to_ix,
                 embedding_dim=None,
                 char_embedding_dim=None,
                 hidden_dim=None,
                 num_layers=None,
                 dropout_keep_prob=0.8,
                 is_cuda=None):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim or EMBEDDING_DIM
        self.char_embedding_dim = char_embedding_dim or CHAR_EMBEDDING_DIM
        self.hidden_dim = hidden_dim or HIDDEN_DIM
        self.tag_to_ix = tag_to_ix
        self.num_layers = num_layers or NUM_LAYERS
        self.dropout_keep_prob = dropout_keep_prob
        self.tagset_size = len(tag_to_ix)
        self.is_cuda = is_cuda if is_cuda is not None else torch.cuda.is_available()

        self.word_encoder = BRNNWordEncoder(self.char_embedding_dim, rnn_type='LSTM')

        if self.is_cuda:
            self.word_encoder = self.word_encoder.cuda()

        self.dropout = nn.Dropout(1 - self.dropout_keep_prob)
        self.lstm = nn.LSTM(self.embedding_dim + self.char_embedding_dim,
                            self.hidden_dim // 2,
                            num_layers=self.num_layers,
                            bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = torch.randn(self.tagset_size, self.tagset_size)
        torch.nn.init.xavier_normal_(self.transitions)

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions[tag_to_ix[START_TAG], :] = -10000
        self.transitions[:, tag_to_ix[STOP_TAG]] = -10000

        if self.is_cuda:
            self.transitions = self.transitions.cuda()

        self.transitions = nn.Parameter(self.transitions)

        self.hidden = self.init_hidden()

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

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas.cuda() if self.is_cuda else init_alphas

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
        seq_len = sentence.size(0)
        # embeds = sentence.view(seq_len, 1, -1)  # [seq_len, batch_size, features]
        lstm_out, self.hidden = self.lstm(sentence, self.hidden)
        lstm_out = lstm_out.view(seq_len, self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.Tensor([0])
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])

        if self.is_cuda:
            score = score.cuda()
            tags = tags.cuda()

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
        forward_var = init_vvars.cuda() if self.is_cuda else init_vvars

        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()
            next_tag_var = next_tag_var.data.cpu().numpy()
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            viterbivars_t = torch.FloatTensor(viterbivars_t)

            if self.is_cuda:
                viterbivars_t = viterbivars_t.cuda()

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

        if self.is_cuda:
            word_embeds = word_embeds.cuda()

        char_embeds = self.word_encoder(sentence).unsqueeze(1)

        sentence_in = torch.cat((word_embeds, char_embeds), dim=-1)
        sentence_in = self.dropout(sentence_in)

        feats = self._get_lstm_features(sentence_in)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        word_embeds = prepare_vec_sequence(sentence, word_to_vec, output='variable')

        if self.is_cuda:
            word_embeds = word_embeds.cuda()

        char_embeds = self.word_encoder(sentence).unsqueeze(1)

        sentence_in = torch.cat((word_embeds, char_embeds), dim=-1)

        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence_in)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
