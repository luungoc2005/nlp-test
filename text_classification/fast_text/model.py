import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import wordpunct_tokenize
from common.wrappers import IModel
from common.utils import pad_sequences
from common.torch_utils import to_gpu
from featurizers.fasttext_featurizer import FastTextFeaturizer
from text_classification.utils.inference import infer_classification_output
from config import MAX_NUM_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH

class FastText(nn.Module):

    def __init__(self, config={}):
        super(FastText, self).__init__()
        self.config = config

        self.max_features = config.get('max_features', MAX_NUM_WORDS)
        self.emb_dropout_prob = config.get('emb_dropout_prob', 0.)
        self.hidden_size = config.get('hidden_size', 100)
        self.h_dropout_prob = config.get('h_dropout_prob', 0.)
        self.n_classes = config.get('num_classes', 10)
        self.embedding_matrix = config.get('embedding_matrix', None)
        self.embedding_dim = config.get('input_shape', (EMBEDDING_DIM,))[-1]

        self.embedding = nn.EmbeddingBag(self.max_features, self.embedding_dim)
        if self.embedding_matrix is not None:
            self.embedding.from_pretrained(self.embedding_matrix)
        self.embedding.requires_grad = False

        self.emb_dropout = nn.Dropout(self.emb_dropout_prob)
        self.i2h = nn.Linear(self.embedding_dim, self.hidden_size, bias=False)
        self.h_dropout = nn.Dropout(self.h_dropout_prob)
        self.h2o = nn.Linear(self.hidden_size, self.n_classes)

        self.init_hidden()

    def init_hidden(self):
        pass

    def forward(self, X):
        """
        X: vector of (batch, token indexes)
        """
        embs = self.embedding(X)
        embs = self.emb_dropout(embs)

        logits = self.i2h(embs)
        logits = self.h_dropout(logits)
        logits = self.h2o(logits)

        if self.training:
            return logits
        else:
            return F.softmax(logits, dim=1)

class FastTextWrapper(IModel):

    def __init__(self, config={}, *args, **kwargs):
        super(FastTextWrapper, self).__init__(
            model_class=FastText, 
            config=config, 
            featurizer=FastTextFeaturizer(),
            *args, **kwargs
        )
        self.config = config
        self.topk = config.get('top_k', 5)
        self.label_encoder = LabelEncoder()

    def get_state_dict(self):
        return {
            'tokenizer': self.tokenizer,
            'label_encoder': self.label_encoder,
        }

    def load_state_dict(self, state_dict):
        config = state_dict['config']

        # re-initialize model with loaded config
        self.topk = config.get('top_k', 5)

        # load label encoder
        self.label_encoder = state_dict['label_encoder']

    def preprocess_output(self, y):
        # One-hot encode outputs
        # Can also use torch.eye() but leaving as numpy until torch achieves performance parity
        # lookup = np.eye(self.num_classes)
        # outputs = np.array([lookup[label] for label in y])
        # return to_gpu(torch.from_numpy(outputs).float())

        return torch.from_numpy(self.label_encoder.transform(y)).long()

    def get_layer_groups(self):
        model = self.model
        return [
            (model.embedding, model.emb_dropout),
            (model.i2h, model.h_dropout),
            model.h2o
        ]
    
    def infer_predict(self, logits, topk=None):
        return infer_classification_output(self, logits, topk)