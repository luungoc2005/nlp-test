import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from sklearn.preprocessing import LabelEncoder
from text_classification.utils.inference import infer_classification_output

from common.wrappers import IModel
from common.utils import dotdict
from common.torch_utils import ACT2FN, to_gpu
from sent_to_vec.masked_lm.bert_model import BertLMWrapper

#TODO: Concat of Max & Mean pool support (requires passing seq length)
class LMClassifier(nn.Module):

    def __init__(self, config=dotdict({
        'encoder_dropout': .2,
        'hidden_size': 512,
        'rnn_layers': 1,
        'pool_hidden_size': 512,
        'pool_layers': 1,
        'pool_act': 'cauchy'
    }), *args, **kwargs):
        super(LMClassifier, self).__init__(*args, **kwargs)
        self.encoder = config.get('encoder', BertLMWrapper)
        
        self.encoder_size = config.get('encoder_size', \
            self.encoder.config.get('hidden_size', 512) \
            if hasattr(self.encoder, 'config') else 512
        )
        self.encoder_dropout = config.get('encoder_dropout', .5)
        self.hidden_size = config.get('hidden_size', 512)
        self.rnn_layers = config.get('rnn_layers', 1)
        self.pool_hidden_size = config.get('pool_hidden_size', 512)
        self.pool_layers = config.get('pool_layers', 2)
        self.pool_act = config.get('pool_act', None)
        if self.pool_act is not None:
            self.pool_act = ACT2FN[self.pool_act]

        self.n_classes = config.get('num_classes', 10)

        if self.encoder_dropout > 0:
            self.dropout_e = nn.Dropout2d(self.encoder_dropout)
        else:
            self.dropout_e = None
        
        if self.hidden_size > 0 and self.rnn_layers > 0:
            self.rnn = nn.LSTM(
                self.encoder_size, 
                self.hidden_size,
                self.rnn_layers, 
                batch_first=True, 
                bidirectional=True
            )
        else:
            self.rnn = None

        if self.hidden_size > 0:
            self.pool_input_size = self.hidden_size * 2 # directions
        else:
            self.pool_input_size = self.encoder_size

        if self.pool_hidden_size > 0 and self.pool_layers > 0:
            pool_layers_list = []
            pool_layers_list.append(
                nn.Linear(self.pool_input_size, self.pool_hidden_size)
            )
            if self.pool_layers > 1:
                for idx in range(self.pool_layers - 1):
                    pool_layers_list.append(
                        nn.Linear(self.pool_hidden_size, self.pool_hidden_size)
                    )
                    
            self.pooler = nn.ModuleList(pool_layers_list)

            self.classifier = nn.Linear(self.pool_hidden_size, self.n_classes, bias=False)
        else:
            self.pooler = None

            self.classifier = nn.Linear(self.pool_input_size, self.n_classes, bias=False)

    def forward(self, seq_batch):
        _, sequence_output = self.encoder(seq_batch)
        
        if self.dropout_e is not None:
            sequence_output = sequence_output.permute(0, 2, 1) # convert to [batch, channels, time]
            sequence_output = self.dropout_e(sequence_output)
            sequence_output = sequence_output.permute(0, 2, 1) # back to [batch, time, channels]

        if self.rnn is not None:
            sequence_output = self.rnn(sequence_output)[0]

        # pooling
        max_pool = torch.max(sequence_output, 1)[0]
        if max_pool.ndimension() == 3:
            max_pool = max_pool.squeeze(1)
        
        pooled = max_pool
        if self.pooler is not None:
            for layer in self.pooler:
                pooled = layer(pooled)
                if self.pool_act is not None:
                    pooled = self.pool_act(pooled)
        
        pooled = self.classifier(pooled)

        batch_size = pooled.size(0)
        inhibited_channel = to_gpu(torch.zeros((batch_size, 1)))
        
        with_inhibited = torch.cat((pooled, inhibited_channel), dim=1)

        return with_inhibited, max_pool

class LMClassifierWrapper(IModel):

    def __init__(self, config={}):
        super(LMClassifierWrapper, self).__init__(
            model_class=LMClassifier, 
            config=config
        )
        self.config = config

        self.topk = config.get('top_k', 5)

        self.label_encoder = LabelEncoder()

    def get_state_dict(self):
        return {
            'label_encoder': self.label_encoder,
        }

    def load_state_dict(self, state_dict):
        # re-initialize model with loaded config
        self.model = self.init_model()
        self.model.set_params(state_dict['state_dict'])

        # load label encoder
        self.label_encoder = state_dict['label_encoder']

    def preprocess_output(self, y):
        # One-hot encode outputs
        # Can also use torch.eye() but leaving as numpy until torch achieves performance parity
        # lookup = np.eye(self.num_classes)
        # outputs = np.array([lookup[label] for label in y])
        # return torch.from_numpy(outputs).float()

        return torch.from_numpy(self.label_encoder.transform(y)).long()

    def infer_predict(self, logits, topk=None):
        return infer_classification_output(self, logits, topk)