import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LM_VOCAB_SIZE, LM_HIDDEN_DIM, LM_SEQ_LEN, LM_CHAR_SEQ_LEN, START_TAG, STOP_TAG, UNK_TAG, MASK_TAG
from common.modules import LockedDropout, WeightDrop
from common.splitcross import SplitCrossEntropyLoss
from common.wrappers import IModel
from common.torch_utils import to_gpu
from featurizers.basic_featurizer import BasicFeaturizer
from common.splitcross import SplitCrossEntropyLoss

from sent_to_vec.masked_lm.densenet import DenseNet
from sent_to_vec.masked_lm.aggregator import Aggregator

from typing import Union, Iterable, Tuple

class PervasiveAttnLanguageModel(nn.Module):

    def __init__(self, config):
        super(PervasiveAttnLanguageModel, self).__init__()
        self.config = config

        self.tie_weights = config.get('tie_weights', True)
        self.embedding_dim = config.get('embedding_dim', LM_HIDDEN_DIM)
        self.dropout_emb = config.get('emb_dropout', .2)
        self.dropout_net = config.get('net_dropout', .2)
        self.num_words = config.get('num_words', LM_VOCAB_SIZE)
        self.n_layers = config.get('n_layers', 6)
        self.use_adasoft = config.get('use_adasoft', True)
        self.adasoft_cutoffs = config.get('adasoft_cutoffs', [LM_VOCAB_SIZE // 2, LM_VOCAB_SIZE // 2])

        self.encoder = nn.Embedding(
            self.num_words, self.embedding_dim
        )
        self.input_channels = self.embedding_dim * 2

        self.net = DenseNet(
            self.input_channels,
            {
                'growth_rate': 32,
                'num_layers': [20],
                'kernels': [3],
                'divde_channels': 2,
                'normalize_channels': 0,
                'dilation': 1,
                'groups': 1,
                'layer_type': 'regular',
                'transition_type': 1,
                'bias': 0,
                'gated': 0,
                'weight_norm': 0,
                'init_weights': 0,
                'conv_dropout': self.dropout_net,
                'efficient': 1
            }
        )

        self.aggregator = Aggregator(
            self.net.output_channels,
            self.embedding_dim if self.tie_weights else self.hidden_dim,
            {
                'mode': 'max',
                'first_aggregator': 'max',
                'attention_dropout': .2,
                'scale_ctx': 1,
                'nonlin': 'none',
                'mapping': 'linear',
                'map_embeddings': 'none'
            }
        )

        self.decoder = nn.Linear(
            self.embedding_dim if self.tie_weights else self.hidden_dim, 
            self.num_words
        )
        self.adasoft = None

        # Weight tying
        if self.tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def merge(self, src_emb, trg_emb):
        return torch.cat((src_emb, trg_emb), dim=3)

    def _expand(self, tensor, dim, reps):
        # Expand 4D tensor in the source or the target dimension
        if dim == 1:
            return tensor.repeat(1, reps, 1, 1)
            # return tensor.expand(-1, reps, -1, -1)
        if dim == 2:
            return tensor.repeat(1, 1, reps, 1)
            # return tensor.expand(-1, -1, reps, -1)
        else:
            raise NotImplementedError

    def _forward(self, X, src_lengths=None, track=False):
        X = X.permute(0, 3, 1, 2)
        X = self.net(X)
        if track:
            X, attn = self.aggregator(X, src_lengths, track=True)
            return X, attn
        X = self.aggregator(X, src_lengths, track=track)
        return X

    def forward(self,
        x_input: Union[torch.LongTensor, torch.cuda.LongTensor], 
        hidden: Union[torch.FloatTensor, torch.cuda.FloatTensor] = None, 
        training: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        training = training or self.training

        src_emb = self.encoder(x_input).permute(1, 0, 2)
        trg_emb = src_emb.clone()
        # trg_emb = self.trg_embedding(data_trg)
        Ts = src_emb.size(1)  # source sequence length
        # Tt = trg_emb.size(1)  # target sequence length
        # 2d grid:
        # src_emb = self._expand(src_emb.unsqueeze(1), 1, Tt)
        # trg_emb = self._expand(trg_emb.unsqueeze(2), 2, Ts)

        src_emb = self._expand(src_emb.unsqueeze(1), 1, Ts)
        trg_emb = self._expand(trg_emb.unsqueeze(2), 2, Ts)

        X = self.merge(src_emb, trg_emb)
        # del src_emb, trg_emb
        # X = self._forward(X, data_src['lengths'])
        X = self._forward(X, None)
        logits = self.decoder(X).permute(1, 0, 2)
        # return logits
        return logits, None, None, None


class PervasiveAttnLanguageModelWrapper(IModel):

    def __init__(self, config=dict(), *args, **kwargs):
        featurizer_config = config
        featurizer_config['append_sos_eos'] = True
        featurizer_config['featurizer_reserved_tokens'] = [START_TAG, STOP_TAG, UNK_TAG, MASK_TAG]

        super(PervasiveAttnLanguageModelWrapper, self).__init__(
            model_class=PervasiveAttnLanguageModel, 
            config=config, 
            featurizer=BasicFeaturizer(featurizer_config),
            *args, **kwargs
        )

        self.seq_len = config.get('seq_len', LM_SEQ_LEN)
        self.config = config

    def on_model_init(self):
        model = self._model

    # def repackage_hidden(self, h) -> Union[torch.Tensor, Tuple]:
    #     if torch.is_tensor(h):
    #         return to_gpu(h.detach())
    #     else:
    #         return tuple(self.repackage_hidden(v) for v in h)
