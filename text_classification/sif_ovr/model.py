from common.keras_preprocessing import Tokenizer
from common.wrappers import IModel
from common.torch_utils import to_gpu
from sent_to_vec.sif.encoder import SIF_embedding
from common.utils import word_to_vec
from config import MAX_NUM_WORDS, EMBEDDING_DIM
from nltk.tokenize import wordpunct_tokenize
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class OvrClassifier(nn.Module):

    def __init__(self, config):
        super(OvrClassifier, self).__init__()

        self.input_dim = config.get('input_dim', EMBEDDING_DIM)
        self.hidden_size = config.get('hidden_size', 0)
        self.h_dropout_prob = config.get('h_dropout_prob', 0.)
        self.num_classes = config.get('num_classes', 10)

        self.classifiers = list()
        for ix in range(self.num_classes):
            if self.hidden_size == 0:
                clf = to_gpu(nn.Sequential(
                    nn.Dropout(self.h_dropout_prob),
                    nn.Linear(self.input_dim, 1)
                ))
            else:
                clf = to_gpu(nn.Sequential(
                    nn.Linear(self.input_dim, self.hidden_size),
                    nn.Dropout(self.h_dropout_prob),
                    nn.Sigmoid(),
                    nn.Linear(self.hidden_size, 1)
                ))
            self.classifiers.append(clf)
    
    def forward(self, embs):
        batch_size = embs.size(0)
        buffer = torch.zeros(batch_size, self.num_classes).float()
        for ix, clf in enumerate(self.classifiers):
            logits = clf(embs)
            buffer[:,ix] = logits[:,0]
        return buffer

    def binary_crossentropy_loss(self, tokens, targets):
        loss = None
        for idx in range(self.num_classes):
            cls_target = (targets == idx).float()
            cls_tokens = tokens[:,idx]
            cls_loss = F.binary_cross_entropy_with_logits(cls_tokens, cls_target)
            
            if loss is None:
                loss = cls_loss
            else:
                loss += cls_loss

        return loss

class OvrClassifierWrapper(IModel):

    def __init__(self, config):
        super(OvrClassifierWrapper, self).__init__(
            model_class=OvrClassifier, 
            config=config
        )

        self.tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        self.num_words = config.get('num_words', MAX_NUM_WORDS)
        self.num_classes = config.get('num_classes', 10)
        
        self.tokenize_fn = wordpunct_tokenize

    def get_state_dict(self):
        return {
            'tokenizer': self.tokenizer,
            'config': self.model.config,
            'state_dict': self.model.get_params(),
        }

    def load_state_dict(self, state_dict):
        config = state_dict['config']

        # re-initialize model with loaded config
        self.model = self.init_model()
        self.model.set_params(state_dict['state_dict'])

        # load tokenizer
        self.tokenizer = state_dict['tokenizer']

    def preprocess_input(self, X):
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

        tokens = [self.tokenize_fn(sent) for sent in X]
        tokens = self.tokenizer.texts_to_sequences(tokens)
        tfidf_matrix = self.tokenizer.sequences_to_matrix(tokens, mode='tfidf')
        
        maxlen = max([len(sent) for sent in tokens])
        tfidf_weights = np.zeros((len(tokens), maxlen))
        for i, seq in enumerate(tokens):
            for j, token in enumerate(seq):
                if token < self.tokenizer.num_words:
                    tfidf_weights[i][j] = tfidf_matrix[i][token]
        
        # convert from token back to texts
        # this is to guarantee that tfidf matrix and X has the same length (with oov words ommited)
        embs = word_to_vec(self.tokenizer.sequences_to_texts(tokens))

        sif_emb = SIF_embedding(embs, tfidf_weights, rmpc=0)

        return torch.from_numpy(sif_emb).float()

    def preprocess_output(self, y):
        # One-hot encode outputs
        # Can also use torch.eye() but leaving as numpy until torch achieves performance parity
        # lookup = np.eye(self.num_classes)
        # outputs = np.array([lookup[label] for label in y])
        # return torch.from_numpy(outputs).float()

        return torch.from_numpy(y).long()
