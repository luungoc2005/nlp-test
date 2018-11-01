import torch
import torch.nn as nn
from config import LM_VOCAB_SIZE, LM_HIDDEN_DIM, LM_SEQ_LEN, LM_CHAR_SEQ_LEN, START_TAG, STOP_TAG, UNK_TAG, EMPTY_TAG, MASK_TAG
from common.modules import LockedDropout, WeightDrop
from common.wrappers import IModel
from common.torch_utils import to_gpu
from featurizers.basic_featurizer import BasicFeaturizer
from typing import Union, Iterable, Tuple

class BiRNNLanguageModel(nn.Module):

    def __init__(self, config):
        super(BiRNNLanguageModel, self).__init__()
        self.config = config

        self.embedding_dim = config.get('embedding_dim', LM_HIDDEN_DIM)
        self.dropout_emb = config.get('emb_dropout', .2)
        self.dropout_i = config.get('lock_drop', .5)
        self.dropout_h = config.get('h_dropout', .5)
        self.wdrop = config.get('wdrop', 0)
        self.num_words = config.get('num_words', LM_VOCAB_SIZE)
        self.rnn_type = config.get('rnn_type', 'SRU')
        self.n_layers = config.get('n_layers', 6)
        self.dropout_rnn = config.get('rnn_dropout', .2)
        self.highway_bias = config.get('highway_bias', -3)

        assert self.rnn_type in ['LSTM', 'GRU', 'SRU']

        self.encoder = nn.Embedding(
            self.num_words, self.embedding_dim
        )
        self.lockdrop = to_gpu(LockedDropout())

        # for the mean time weight drop is broken
        if self.rnn_type == 'LSTM':
            self.rnns = nn.ModuleList([
                nn.LSTM(
                    self.embedding_dim, 
                    self.embedding_dim // 2,
                    bidirectional=True
                )
                for layer_ix in range(self.n_layers)
            ])
        elif self.rnn_type == 'GRU':
            self.rnns = nn.ModuleList([
                nn.GRU(
                    self.embedding_dim, 
                    self.embedding_dim // 2,
                    bidirectional=True
                )
                for layer_ix in range(self.n_layers)
        ])
        else:
            from sru import SRU
            self.rnns = nn.ModuleList([
                to_gpu(SRU(
                    self.embedding_dim, 
                    self.embedding_dim // 2,
                    num_layers=1,
                    rnn_dropout=self.dropout_rnn,
                    dropout=self.wdrop,
                    rescale=False,
                    highway_bias=self.highway_bias,
                    use_tanh=0,
                    bidirectional=True,
                    v1=True
                )) 
                for layer_ix in range(self.n_layers)
            ])

        self.decoder = nn.Linear(self.embedding_dim, self.num_words)

        # Weight tying
        self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size) -> Iterable[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        if self.rnn_type == 'LSTM':
            return [
                (to_gpu(torch.zeros(
                    2, 
                    batch_size, 
                    self.embedding_dim // 2
                )),
                to_gpu(torch.zeros(
                    2, 
                    batch_size, 
                    self.embedding_dim // 2
                )))
                for l in range(self.n_layers)
            ]
        elif self.rnn_type == 'SRU' or self.rnn_type == 'GRU':
            return [
                to_gpu(torch.zeros(
                    1, 
                    batch_size, 
                    self.embedding_dim
                ))
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

        X = to_gpu(torch.nn.functional.embedding(words, masked_embed_weight,
            padding_idx, embed.max_norm, embed.norm_type,
            embed.scale_grad_by_freq, embed.sparse
        ))
        
        return X

    def forward(self, x_input, hidden=None, return_raws=False) -> \
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

        if hidden is None:
            # Try to determine batch size and generate hidden from input
            batch_size = x_input.size(1)
            hidden = self.init_hidden(batch_size)

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

        output = self.lockdrop(raw_output, self.dropout_h)
        outputs.append(output)

        result = output.view(output.size(0) * output.size(1), output.size(2))

        if return_raws:
            return result, raw_hiddens, raw_outputs, outputs
        else:
            return result, raw_hiddens

class BiLanguageModelWrapper(IModel):

    def __init__(self, config=dict(), *args, **kwargs):
        featurizer_config = config
        featurizer_config['append_sos_eos'] = True
        featurizer_config['featurizer_reserved_tokens'] = [EMPTY_TAG, START_TAG, STOP_TAG, UNK_TAG, MASK_TAG]

        super(BiLanguageModelWrapper, self).__init__(
            model_class=BiRNNLanguageModel, 
            config=config, 
            featurizer=BasicFeaturizer(featurizer_config),
            *args, **kwargs
        )

        self.seq_len = config.get('seq_len', LM_SEQ_LEN)
        self.config = config

    def generate(self, n_tokens, temperature=1.):
        self.model.eval()
        self.hidden = self.model.init_hidden(1)

        seed = torch.rand(1, 1).mul(n_tokens).long()
        retstr = []
        # retidx = []

        with torch.no_grad():
            for ix in range(n_tokens):
                seed = to_gpu(seed)

                output, self.hidden = self.model(seed, self.hidden)
                word_weights = output.squeeze().data.div(temperature).exp().cpu()
                
                # filter out inf and negative probabilities
                word_weights[word_weights == float("Inf")] = 0
                word_weights[word_weights < 0] = 0

                word_idx = torch.multinomial(word_weights, 1)[0]
                seed.data.fill_(word_idx)

                word_idx = int(word_idx)
                word = self.featurizer.tokenizer.ix_to_word.get(word_idx, '')
                retstr += [word]
                # retidx += [word_idx]

        self.model.train()

        return ' '.join(retstr)

    def repackage_hidden(self, h) -> Union[torch.Tensor, Tuple]:
        if torch.is_tensor(h):
            return to_gpu(h.detach())
        else:
            return tuple(self.repackage_hidden(v) for v in h)
