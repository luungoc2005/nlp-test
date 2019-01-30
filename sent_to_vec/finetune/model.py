import torch
import torch.nn as nn
from config import LM_VOCAB_SIZE, LM_HIDDEN_DIM, LM_SEQ_LEN
from common.modules import LockedDropout, WeightDrop
from common.wrappers import IModel
from common.torch_utils import to_gpu
from sent_to_vec.awd_lm.model import LanguageModelWrapper
from featurizers.basic_featurizer import BasicFeaturizer
from typing import Union, Iterable, Tuple

class DoubleClassificationHead(nn.Module):

    def __init__(self, config):
        super(DoubleClassificationHead, self).__init__()
        self.config = config

        self.fc_dim = config.get('fc_dim', 512)
        self.nonlinear_fc = config.get('nonlinear_fc', False)
        self.dropout_fc = config.get('dropout_fc', .5)
        self.fc_output_dim = config.get('n_classes', 2)
        self.fc_dist_features = config.get('fc_dist_features', False)
        self.fc_input_dim = config.get(
            'fc_input_dim', 
            config.get('embedding_dim', LM_HIDDEN_DIM)
        )
        self.encoder = None

        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dropout_fc),
                nn.Linear(self.fc_input_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dropout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dropout_fc),
                nn.Linear(self.fc_dim, self.fc_output_dim),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.fc_input_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_output_dim)
            )

    def set_encoder(self, encoder):
        self.encoder = encoder

    def forward(self, s1, s2):
        assert self.encoder is not None, 'set_encoder must be called before forward()'
    
        u = self.encoder(s1)
        v = self.encoder(s2)

        if self.fc_dist_features:
            features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        else:
            features = torch.cat((u, v), 1)
        
        return self.classifier(features)

class SingleClassificationHead(nn.Module):

    def __init__(self, config):
        super(SingleClassificationHead, self).__init__()
        self.config = config

        self.fc_dim = config.get('fc_dim', 512)
        self.nonlinear_fc = config.get('nonlinear_fc', False)
        self.dropout_fc = config.get('dropout_fc', .5)
        self.fc_output_dim = config.get('n_classes', 2)
        self.fc_dist_features = config.get('fc_dist_features', False)
        self.fc_input_dim = config.get(
            'fc_input_dim', 
            config.get('embedding_dim', LM_HIDDEN_DIM)
        )
        self.encoder = None

        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dropout_fc),
                nn.Linear(self.fc_input_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dropout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dropout_fc),
                nn.Linear(self.fc_dim, self.fc_output_dim),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.fc_input_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_output_dim)
            )

    def set_encoder(self, encoder):
        self.encoder = encoder

    def forward(self, sent):
        assert self.encoder is not None, 'set_encoder must be called before forward()'
    
        feats = self.encoder(sent)
        
        return self.classifier(feats)

class ClassificationHeadWrapper(IModel):

    def __init__(self, config=dict(), *args, **kwargs):
        featurizer_config = config
        featurizer_config['append_sos_eos'] = True

        super(ClassificationHeadWrapper, self).__init__(
            model_class=ClassificationHeadWrapper, 
            config=config, 
            featurizer=BasicFeaturizer(featurizer_config),
            *args, **kwargs
        )

        self.seq_len = config.get('seq_len', LM_SEQ_LEN)
        self.n_classes = config.get('n_classes', 2)
        self.config = config
        self.encoder = None

    def set_encoder(self, encoder):
        self.encoder = encoder
        
    def on_model_init(self):
        super().on_model_init()
        if self.encoder is not None:
            self.model.set_encoder(self.encoder)