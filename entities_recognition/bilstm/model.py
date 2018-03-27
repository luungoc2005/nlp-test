import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import START_TAG, STOP_TAG, EMBEDDING_DIM, HIDDEN_DIM

from common.utils import letterToIndex, n_letters, prepare_vec_sequence, word_to_vec, to_scalar, argmax, prepare_sequence, log_sum_exp

class BiLSTMWordEncoder(nn.Module):

    def __init__(self, 
                 hidden_dim = None,
                 letters_dim = None,
                 dropout_keep_prob = 1):
        super(BiLSTMWordEncoder, self).__init__()

        self.hidden_dim = hidden_dim or EMBEDDING_DIM
        self.letters_dim = letters_dim or n_letters
        self.dropout_keep_prob = dropout_keep_prob

        self.rnn = nn.LSTM(n_letters,
                           self.hidden_dim // 2,
                           num_layers=1,
                           dropout=1-self.dropout_keep_prob,
                           bidirectional=True)

    def _letter_to_array(self, letter):
        ret_val = np.zeros(1, self.letters_dim)
        ret_val[0][letterToIndex(letter)] = 1
        return ret_val

    def _word_to_array(self, word):
        ret_val = np.zeros(len(word), 1, self.letters_dim)
        for li, letter in enumerate(word):
            ret_val[li][0][letterToIndex(letter)] = 1
        return ret_val

    def _process_sentence(self, sentence):
        word_lengths = np.array([len(word) for word in sentence])
        max_len = np.max(word_lengths)
        words_batch = np.zeros((max_len, len(sentence), n_letters))

        for i in range(len(sentence)):
            for li, letter in enumerate(sentence[i]):
                words_batch[li][0][letterToIndex(letter)] = 1.
        
        words_batch = Variable(torch.from_numpy(words_batch).float())
        return words_batch, word_lengths
    
    def forward(self, sentence):
        words_batch, word_lengths = self._process_sentence(sentence)

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

        return embeds

class BiLSTM_CRF(nn.Module):

    def __init__(self, 
                 tag_to_ix,
                 embedding_dim = None, 
                 hidden_dim = None,
                 dropout_keep_prob = 0.5):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim or EMBEDDING_DIM
        self.hidden_dim = hidden_dim or HIDDEN_DIM
        self.tag_to_ix = tag_to_ix
        self.dropout_keep_prob = dropout_keep_prob
        self.tagset_size = len(tag_to_ix)

        self.word_encoder = BiLSTMWordEncoder(self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim * 2, 
                            self.hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(1 - self.dropout_keep_prob)

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
        return (Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                Variable(torch.randn(2, 1, self.hidden_dim // 2)))

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
        embeds = sentence.view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_out = self.dropout(lstm_out)
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
        sentence_in = torch.cat([word_embeds, char_embeds], dim=1)

        feats = self._get_lstm_features(sentence_in)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        word_embeds = prepare_vec_sequence(sentence, word_to_vec, output='variable')
        char_embeds = self.word_encoder(sentence)
        sentence_in = torch.cat([word_embeds, char_embeds], dim=1)

        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence_in)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq