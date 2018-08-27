import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings
from common.wrappers import IModel
from common.utils import pad_sequences, prepare_vec_sequence, wordpunct_space_tokenize, word_to_vec
from common.torch_utils import to_gpu
from common.keras_preprocessing import Tokenizer
from config import MAX_NUM_WORDS, EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, START_TAG, STOP_TAG
from common.modules import BRNNWordEncoder

class SequenceTagger(nn.Module):

    def __init__(self, config):
        super(SequenceTagger, self).__init__()
        self.config = config

        self.embedding_dim = config.get('embedding_dim', EMBEDDING_DIM)
        self.char_embedding_dim = config.get('char_embedding_dim', CHAR_EMBEDDING_DIM)
        self.emb_dropout_prob = config.get('emb_dropout_prob', .2)
        self.hidden_size = config.get('hidden_size', HIDDEN_DIM)
        self.num_layers = config.get('num_layers', NUM_LAYERS)
        self.h_dropout_prob = config.get('h_dropout_prob', 0.)

        self.tag_to_ix = config.get('tag_to_ix', {START_TAG: 0, STOP_TAG: 1})
        self.tagset_size = len(self.tag_to_ix)

        self.word_encoder = BRNNWordEncoder(self.char_embedding_dim, rnn_type='LSTM')
        self.emb_dropout = nn.Dropout(emb_dropout_prob)
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

        self.transitions = to_gpu(self.transitions)

        self.transitions = nn.Parameter(self.transitions)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden_0 = torch.randn(self.num_layers * 2, 1, self.hidden_dim // 2)
        hidden_1 = torch.randn(self.num_layers * 2, 1, self.hidden_dim // 2)

        return to_gpu(hidden_0), to_gpu(hidden_1)

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = to_gpu(init_alphas)

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
        sentence = self.emb_dropout(sentence)
        lstm_out, self.hidden = self.lstm(sentence, self.hidden)
        lstm_out = lstm_out.view(seq_len, self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = to_gpu(torch.Tensor([0]))
        tags = to_gpu(torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags]))

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
        forward_var = to_gpu(init_vvars)

        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()
            next_tag_var = next_tag_var.data.cpu().numpy()
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            viterbivars_t = torch.FloatTensor(viterbivars_t)

            viterbivars_t = to_gpu(viterbivars_t)

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
        word_embeds = to_gpu(prepare_vec_sequence(sentence, word_to_vec, output='tensor'))

        char_embeds = self.word_encoder(sentence).unsqueeze(1)

        sentence_in = torch.cat((word_embeds, char_embeds), dim=-1)
        sentence_in = self.dropout(sentence_in)

        feats = self._get_lstm_features(sentence_in)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return feats, forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        word_embeds = to_gpu(prepare_vec_sequence(sentence, word_to_vec, output='tensor'))

        char_embeds = self.word_encoder(sentence).unsqueeze(1)

        sentence_in = torch.cat((word_embeds, char_embeds), dim=-1)

        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence_in)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


class SequenceTaggerWrapper(IModel):

    def __init__(self, config):
        super(SequenceTaggerWrapper, self).__init__(
            model_class=SequenceTagger,
            config=config
        )
        self.tokenizer = wordpunct_space_tokenize
        self.tag_to_ix = config.get('tag_to_ix', {START_TAG: 0, STOP_TAG: 1})

    def get_state_dict(self):
        return {
            'config': self.model.config,
            'state_dict': self.model.state_dict(),
        }

    def load_state_dict(self, state_dict):
        config = state_dict['config']

        # re-initialize model with loaded config
        self.model = self._model_class(config)
        self.model.load_state_dict(state_dict['state_dict'])
        self.tag_to_ix = config.get('tag_to_ix', {START_TAG: 0, STOP_TAG: 1})

    def preprocess_input(self, X):
        return self.tokenizer(X[0])
    
    def preprocess_output(self, y):
        return torch.LongTensor([self.tag_to_ix[tag] for tag in y[0].split()])

    def get_layer_groups(self):
        model = self.model
        return [
            (*zip(model.word_encoder.get_layer_groups()), model.emb_dropout),
            model.lstm,
            model.hidden2tag
        ]