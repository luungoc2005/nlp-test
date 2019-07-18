import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from sklearn.preprocessing import LabelEncoder
from text_classification.utils.inference import infer_classification_output

from common.wrappers import IModel
from common.utils import dotdict
from common.torch_utils import ACT2FN, to_gpu
from sent_to_vec.masked_lm.bert_model import BertLMWrapper, BertForMaskedLM

#TODO: Concat of Max & Mean pool support (requires passing seq length)
class LMClassifier(nn.Module):

    def __init__(self, config=dotdict({
        'encoder_dropout': .2,
        'encoder_config': dotdict({
            'num_words': 36000,
            'hidden_size': 576,
            'num_hidden_layers': 6,
            'num_attention_heads': 12,
            'intermediate_size': 1200,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.15,
            'attention_probs_dropout_prob': 0.15,
            'max_position_embeddings': 104,
            'featurizer_seq_len': 104,
            'type_vocab_size': 2,
            'initializer_range': 0.025,
            'use_adasoft': True
        }),
        'hidden_size': 512,
        'pool_hidden_size': 512,
        'pool_layers': 2,
        'pool_act': 'relu',
        'pool_last_act': 'cauchy'
    }), *args, **kwargs):
        super(LMClassifier, self).__init__(*args, **kwargs)
        self.encoder_config = config.get('encoder_config', dotdict({}))
        print(self.encoder_config)
        self.encoder = BertForMaskedLM(self.encoder_config)
        
        self.encoder_size = config.get('encoder_size', \
            self.encoder.config.get('hidden_size', 512) \
            if hasattr(self.encoder, 'config') else 512
        )
        self.encoder_dropout = config.get('encoder_dropout', .5)
        self.pool_hidden_size = config.get('pool_hidden_size', 512)
        self.pool_layers = config.get('pool_layers', 2)
        self.pool_act = config.get('pool_act', None)
        self.pool_last_act = config.get('pool_last_act', None)
        if self.pool_act is not None:
            self.pool_act = ACT2FN[self.pool_act]
        if self.pool_last_act is not None:
            self.pool_last_act = ACT2FN[self.pool_last_act]

        self.n_classes = config.get('num_classes', 10)

        if self.encoder_dropout > 0:
            self.dropout_e = nn.Dropout2d(self.encoder_dropout)
        else:
            self.dropout_e = None

        self.pool_input_size = self.encoder_size

        if self.pool_hidden_size > 0 and self.pool_layers > 0:
            pool_layers_list = []
            pool_layers_list.append(
                nn.Linear(self.pool_input_size, self.pool_hidden_size)
            )
            if self.pool_layers > 1:
                for _ in range(self.pool_layers - 1):
                    pool_layers_list.append(
                        nn.Linear(self.pool_hidden_size, self.pool_hidden_size)
                    )
                    
            self.pooler = nn.ModuleList(pool_layers_list)

            self.classifier = nn.Linear(self.pool_hidden_size, self.n_classes, bias=False)
        else:
            self.pooler = None

            self.classifier = nn.Linear(self.pool_input_size, self.n_classes, bias=False)

    def forward(self, seq_batch):
        pooled_output = self.encoder(seq_batch)[1]
        
        if self.dropout_e is not None:
            pooled_output = self.dropout_e(pooled_output)

        # pooling
        # pooled_output = torch.max(pooled_output, 1)[0]
        # if pooled_output.ndimension() == 3:
        #     pooled_output = pooled_output.squeeze(1)
        
        if self.pooler is not None:
            for ix, layer in enumerate(self.pooler):
                pooled_output = layer(pooled_output)

                if ix < len(self.pooler) - 1:
                    if self.pool_act is not None:
                        pooled_output = self.pool_act(pooled_output)
                else:
                    if self.pool_last_act is not None:
                        pooled_output = self.pool_last_act(pooled_output)

        pooled_output = self.classifier(pooled_output)

        batch_size = pooled_output.size(0)
        inhibited_channel = to_gpu(torch.zeros((batch_size, 1)))

        with_inhibited = torch.cat((pooled_output, inhibited_channel), dim=1)

        if self.training:
            return with_inhibited, pooled_output
        else:
            return torch.softmax(with_inhibited, dim=1), pooled_output

class LMClassifierWrapper(IModel):

    def __init__(self, 
            config={}, 
            encoder=None, 
            *args, **kwargs
        ):
        super(LMClassifierWrapper, self).__init__(
            model_class=LMClassifier, 
            config=config,
            *args, **kwargs
        )
        self.encoder = encoder
        self.topk = config.get('top_k', 5)
        self.label_encoder = LabelEncoder()
        if self.encoder is not None:
            self.config.update({
                'encoder_config': self.encoder.config
            })

    def get_state_dict(self):
        return {
            'label_encoder': self.label_encoder,
        }

    def load_state_dict(self, state_dict):
        # load label encoder
        self.label_encoder = state_dict['label_encoder']

    def on_model_init(self):
        if self.encoder is not None:
            self._model.encoder = self.encoder._model
            self._featurizer = self.encoder._featurizer
            self.encoder = None

    def preprocess_output(self, y):
        # One-hot encode outputs
        # Can also use torch.eye() but leaving as numpy until torch achieves performance parity
        # lookup = np.eye(self.num_classes)
        # outputs = np.array([lookup[label] for label in y])
        # return torch.from_numpy(outputs).float()

        return torch.from_numpy(self.label_encoder.transform(y)).long()

    def infer_predict(self, logits, topk=None):
        return infer_classification_output(self, logits, topk)