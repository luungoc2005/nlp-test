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

import pyro
from pyro.distributions import Normal, Categorical

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
            predict_fn=self.predict,
            *args, **kwargs
        )
        self.config = config
        self.topk = config.get('top_k', 5)
        self.label_encoder = LabelEncoder()
        self.num_samples = config.get('num_samples', 100)

    def get_state_dict(self):
        return {
            'config': self.config,
            'top_k': self.topk,
            'label_encoder': self.label_encoder
        }

    def load_state_dict(self, state_dict):
        config = state_dict.get('config', dict())
        
        # re-initialize model with loaded config
        self.topk = config.get('top_k', 5)

        # load label encoder
        self.label_encoder = state_dict['label_encoder']

    def init_model(self):
        super().init_model()

        model = self.model

        def pyro_model(X_data, y_data):
            # model.embedding.requires_grad = False
            scale = 1.
            i2h_w_prior = Normal(
                loc=torch.zeros_like(model.i2h.weight), 
                scale=torch.ones_like(model.i2h.weight) * scale
            )
            # i2h_b_prior = Normal(
            #     loc=torch.zeros_like(model.i2h.bias), 
            #     scale=torch.ones_like(model.i2h.bias)
            # )
            h2o_w_prior = Normal(
                loc=torch.zeros_like(model.h2o.weight), 
                scale=torch.ones_like(model.h2o.weight) * scale
            )
            h2o_b_prior = Normal(
                loc=torch.zeros_like(model.h2o.bias), 
                scale=torch.ones_like(model.h2o.bias) * scale
            )
            priors = {
                'i2h.weight': i2h_w_prior, 
                # 'i2h.bias': i2h_b_prior,  
                'h2o.weight': h2o_w_prior, 
                'h2o.bias': h2o_b_prior
            }
            lifted_module = pyro.random_module('module', model, priors)
            lifted_reg_model = lifted_module()

            lhat = F.log_softmax(lifted_reg_model(X_data), dim=1)
            pyro.sample('obs', Categorical(logits=lhat), obs=y_data)

        def pyro_guide(X_data, y_data):
            softplus = nn.Softplus()

            # First layer weight distribution priors
            i2h_w_mu = torch.randn_like(model.i2h.weight)
            i2h_w_sigma = torch.randn_like(model.i2h.weight)
            i2h_w_mu_param = pyro.param("i2h_mu", i2h_w_mu)
            i2h_w_sigma_param = softplus(pyro.param("i2h_sigma", i2h_w_sigma))
            # i2h_w_sigma_param = pyro.param("i2h_sigma", i2h_w_sigma)
            i2h_w_prior = Normal(loc=i2h_w_mu_param, scale=i2h_w_sigma_param)

            # First layer bias distribution priors
            # i2h_b_mu = torch.randn_like(model.i2h.bias)
            # i2h_b_sigma = torch.randn_like(model.i2h.bias)
            # i2h_b_mu_param = pyro.param("i2h_b_mu", i2h_b_mu)
            # i2h_b_sigma_param = softplus(pyro.param("i2h_b_sigma", i2h_b_sigma))
            # i2h_b_prior = Normal(loc=i2h_b_mu_param, scale=i2h_b_sigma_param)
            
            # Output layer weight distribution priors
            h2o_w_mu = torch.randn_like(model.h2o.weight)
            h2o_w_sigma = torch.randn_like(model.h2o.weight)
            h2o_w_mu_param = pyro.param("h2o_w_mu", h2o_w_mu)
            h2o_w_sigma_param = softplus(pyro.param("h2o_w_sigma", h2o_w_sigma))
            # h2o_w_sigma_param = pyro.param("h2o_w_sigma", h2o_w_sigma)
            h2o_w_prior = Normal(loc=h2o_w_mu_param, scale=h2o_w_sigma_param)

            # Output layer bias distribution priors
            h2o_b_mu = torch.randn_like(model.h2o.bias)
            h2o_b_sigma = torch.randn_like(model.h2o.bias)
            h2o_b_mu_param = pyro.param("h2o_b_mu", h2o_b_mu)
            h2o_b_sigma_param = softplus(pyro.param("h2o_b_sigma", h2o_b_sigma))
            # h2o_b_sigma_param = pyro.param("h2o_b_sigma", h2o_b_sigma)
            h2o_b_prior = Normal(loc=h2o_b_mu_param, scale=h2o_b_sigma_param)
            
            priors = {
                'i2h.weight': i2h_w_prior, 
                # 'i2h.bias': i2h_b_prior, 
                'h2o.weight': h2o_w_prior, 
                'h2o.bias': h2o_b_prior
            }
            
            lifted_module = pyro.random_module("module", model, priors)
            
            return lifted_module()

        self.pyro_guide = pyro_guide
        self.pyro_model = pyro_model

    def preprocess_output(self, y):
        # One-hot encode outputs
        # Can also use torch.eye() but leaving as numpy until torch achieves performance parity
        # lookup = np.eye(self.config['num_classes'])
        # outputs = np.array([lookup[label] for label in self.label_encoder.transform(y)])
        # return to_gpu(torch.from_numpy(outputs).float())

        return torch.from_numpy(self.label_encoder.transform(y)).long()

    def get_layer_groups(self):
        model = self.model
        return [
            (model.embedding, model.emb_dropout),
            (model.i2h, model.h_dropout),
            model.h2o
        ]

    # function to encode sentences to vectors
    def encode(self, sents):
        emb = self._featurizer.transform(sents)
        return emb

    def predict(self, X):
        sampled_models = [self.pyro_guide(None, None) for _ in range(self.num_samples)]
        yhats = [model(X).data for model in sampled_models]
        mean = torch.mean(torch.stack(yhats), 0)
        return mean

    def infer_predict(self, logits, topk=None):
        return infer_classification_output(self, logits, topk)