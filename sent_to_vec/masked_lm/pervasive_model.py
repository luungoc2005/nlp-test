import torch
import torch.nn as nn
from config import LM_VOCAB_SIZE, LM_HIDDEN_DIM, LM_SEQ_LEN, LM_CHAR_SEQ_LEN, START_TAG, STOP_TAG, UNK_TAG, MASK_TAG
from common.modules import LockedDropout, WeightDrop
from common.splitcross import SplitCrossEntropyLoss
from common.wrappers import IModel
from common.torch_utils import to_gpu
from featurizers.basic_featurizer import BasicFeaturizer
from typing import Union, Iterable, Tuple
from common.splitcross import SplitCrossEntropyLoss
from typing import Union, List, Iterable

from sent_to_vec.masked_lm.densenet import DenseNet
from sent_to_vec.masked_lm.aggregator import Aggregator

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
                'num_layers': 20,
                'divde_channels': 2,
                'normalize_channels': 0,
                'kernels': 3,
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

    def forward(self, 
        x_input:Union[torch.LongTensor, torch.cuda.LongTensor], 
        hidden:Union[torch.FloatTensor, torch.cuda.FloatTensor] = None, 
        training:bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        training = training or self.training

        if training:
            emb = self.embedded_dropout(
                self.encoder, 
                x_input, 
                self.dropout_emb if self.training else 0
            )
            emb = self.lockdrop(emb, self.dropout_i)
        else:
            emb = self.encoder(x_input)

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
                if training:
                    raw_output = self.lockdrop(raw_output, self.dropout_h)
                
                outputs.append(raw_output)

        if training:
            output = self.lockdrop(raw_output, self.dropout_h)
        else:
            output = raw_output

        outputs.append(output)

        # decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))

        if training:
            # logprob = self.adasoft.\
            #     logprob(
            #         self.decoder.weight, 
            #         self.decoder.bias, 
            #         output.view(output.size(0) * output.size(1), output.size(2))
            #     )
            return output, raw_hiddens, raw_outputs, outputs
        else:
            decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
            return decoded, raw_hiddens

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
        if model is not None and model.rnn_type != 'QRNN':
            for rnn in model.rnns:
                if issubclass(type(rnn), nn.RNNBase):
                    rnn.flatten_parameters()

    # def repackage_hidden(self, h) -> Union[torch.Tensor, Tuple]:
    #     if torch.is_tensor(h):
    #         return to_gpu(h.detach())
    #     else:
    #         return tuple(self.repackage_hidden(v) for v in h)
