import torch
import torch.nn as nn
from config import LM_VOCAB_SIZE, LM_EMBEDDING_DIM, LM_HIDDEN_DIM, LM_SEQ_LEN
from common.modules import LockedDropout
from common.wrappers import IModel
from common.torch_utils import to_gpu
from featurizers.basic_featurizer import BasicFeaturizer
from typing import Union, Iterable, Tuple

class RNNLanguageModel(nn.Module):

    def __init__(self, config):
        super(RNNLanguageModel, self).__init__()
        self.config = config

        self.embedding_dim = config.get('embedding_dim', LM_EMBEDDING_DIM)
        self.dropout_emb = config.get('emb_dropout', .1)
        self.dropout_i = config.get('lock_drop', .5)
        self.dropout_h = config.get('h_dropout', .5)
        self.num_words = config.get('num_words', LM_VOCAB_SIZE)
        self.rnn_type = config.get('rnn_type', 'SRU')
        self.hidden_size = config.get('hidden_size', LM_HIDDEN_DIM)
        self.n_layers = config.get('n_layers', 6)
        self.dropout_rnn = config.get('rnn_dropout', .2)

        assert self.rnn_type in ['LSTM', 'GRU', 'SRU']

        self.encoder = nn.Embedding(
            self.num_words, self.embedding_dim
        )
        self.lockdrop = LockedDropout()

        if self.rnn_type == 'LSTM':
            self.rnns = [nn.LSTM(
                self.embedding_dim if layer_ix == 0 else self.hidden_size // 2, 
                self.hidden_size if layer_ix != self.n_layers - 1 else self.embedding_dim // 2,
                dropout=self.dropout_rnn,
                bidirectional=True
            ) for layer_ix in range(self.n_layers)]
        elif self.rnn_type == 'GRU':
            self.rnns = [nn.GRU(
                self.embedding_dim if layer_ix == 0 else self.hidden_size // 2, 
                self.hidden_size if layer_ix != self.n_layers - 1 else self.embedding_dim // 2,
                dropout=self.dropout_rnn,
                bidirectional=True
            ) for layer_ix in range(self.n_layers)]
        else:
            from sru import SRU
            self.rnns = [SRU(
                self.embedding_dim if layer_ix == 0 else self.hidden_size // 2, 
                self.hidden_size if layer_ix != self.n_layers - 1 else self.embedding_dim // 2,
                rnn_dropout=self.dropout_rnn,
                bidirectional=True,
                v1=True
            ) for layer_ix in range(self.n_layers)]

        self.decoder = nn.Linear(self.embedding_dim, self.num_words)

        # Weight tying
        self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size) -> Iterable[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [
                (weight.new(
                    2, 
                    batch_size, 
                    self.hidden_size // 2 if l != self.n_layers - 1 else self.embedding_dim // 2
                ).zero_(),
                weight.new(
                    2, 
                    batch_size, 
                    self.hidden_size // 2 if l != self.n_layers - 1 else self.embedding_dim // 2
                ).zero_())
                for l in range(self.n_layers)
            ]
        elif self.rnn_type == 'SRU' or self.rnn_type == 'GRU':
            return [
                weight.new(
                    2, 
                    batch_size, 
                    self.hidden_size // 2 if l != self.n_layers - 1 else self.embedding_dim // 2
                ).zero_()
                for l in range(self.n_layers)
            ]

    def embedded_dropout(self, embed, words, dropout=0.1, scale=None):
        if dropout:
            mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
            masked_embed_weight = mask * embed.weight
        else:
            masked_embed_weight = embed.weight
        if scale:
            masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

        padding_idx = embed.padding_idx
        if padding_idx is None:
            padding_idx = -1

        X = torch.nn.functional.embedding(words, masked_embed_weight,
            padding_idx, embed.max_norm, embed.norm_type,
            embed.scale_grad_by_freq, embed.sparse
        )
        
        return X

    def forward(self, x_input, hidden, return_raws=False) -> \
        Union[
            Tuple[torch.Tensor, torch.Tensor], 
            Tuple[torch.Tensor, torch.Tensor, Iterable[torch.Tensor], Iterable[torch.Tensor]]
        ]:
        emb = self.embedded_dropout(
            self.encoder, 
            x_input, 
            self.dropout_emb if self.training else 0
        )
        emb = self.lockdrop(emb, self.dropout_i)

        raw_output = emb

        raw_hiddens = []
        raw_outputs = []
        outputs = []

        for idx, rnn in enumerate(self.rnns):
            raw_output, current_h = rnn(raw_output, hidden[idx])
            
            raw_hiddens.append(current_h)
            raw_outputs.append(raw_output)

            if idx != self.n_layers - 1:
                raw_output = self.lockdrop(raw_output, self.dropout_h)
                outputs.append(raw_output)
            
        hidden = current_h

        output = self.lockdrop(raw_output, self.dropout_h)
        outputs.append(output)

        result = output.view(output.size(0) * output.size(1), output.size(2))

        if return_raws:
            return result, hidden, raw_outputs, outputs
        else:
            return result, hidden

class LanguageModelWrapper(IModel):

    def __init__(self, config=dict(), *args, **kwargs):
        featurizer_config = config
        featurizer_config['append_sos_eos'] = True

        super(LanguageModelWrapper, self).__init__(
            model_class=RNNLanguageModel, 
            config=config, 
            featurizer=BasicFeaturizer(featurizer_config),
            *args, **kwargs
        )

        self.seq_len = config.get('seq_len', LM_SEQ_LEN)
        self.config = config