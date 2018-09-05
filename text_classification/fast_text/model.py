import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings
from nltk.tokenize import wordpunct_tokenize
from common.wrappers import IModel
from common.utils import pad_sequences
from common.torch_utils import to_gpu
from common.keras_preprocessing import Tokenizer
from config import MAX_NUM_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH

class FastText(nn.Module):

    def __init__(self, config):
        super(FastText, self).__init__()
        self.config = config

        self.max_features = config.get('max_features', MAX_NUM_WORDS)
        self.emb_dropout_prob = config.get('emb_dropout_prob', 0.)
        self.hidden_size = config.get('hidden_size', 100)
        self.h_dropout_prob = config.get('h_dropout_prob', 0.)
        self.num_classes = config.get('num_classes', 10)
        self.embedding_matrix = config.get('embedding_matrix', None)
        self.embedding_dim = config.get('embedding_dim', EMBEDDING_DIM)

        self.embedding = nn.EmbeddingBag(self.max_features, self.embedding_dim)
        if self.embedding_matrix is not None:
            self.embedding.from_pretrained(self.embedding_matrix)
        self.embedding.requires_grad = False

        self.emb_dropout = nn.Dropout(self.emb_dropout_prob)
        self.i2h = nn.Linear(self.embedding_dim, self.hidden_size)
        self.h_dropout = nn.Dropout(self.h_dropout_prob)
        self.h2o = nn.Linear(self.hidden_size, self.num_classes)

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

    def __init__(self, config):
        super(FastTextWrapper, self).__init__(
            model_class=FastText, 
            config=config
        )

        self.tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        self._ngrams = config.get('ngrams', 2)
        self._max_features = config.get('num_words', MAX_NUM_WORDS)
        self.num_words = config.get('num_words', MAX_NUM_WORDS)
        self.num_classes = config.get('num_classes', 10)
        self.max_len = config.get('max_len', MAX_SEQUENCE_LENGTH)
        
        token_indice = config.get('token_indice', {})
        self.token_indice = token_indice
        self.indice_token = {token_indice[k]: k for k in token_indice}

        self.tokenize_fn = wordpunct_tokenize

    def get_state_dict(self):
        return {
            'tokenizer': self.tokenizer,
            'config': self.model.config,
            'state_dict': self.model.state_dict(),
            'token_indice': self.token_indice,
            'indice_token': self.indice_token
        }

    def load_state_dict(self, state_dict):
        config = state_dict['config']

        # re-initialize model with loaded config
        self.model = self._model_class(config)
        self.model.load_state_dict(state_dict['state_dict'])

        # load tokenizer
        self.tokenizer = state_dict['tokenizer']

        self.token_indice = state_dict['token_indice']
        self.indice_token = state_dict['indice_token']
        
        self._ngrams = config.get('ngrams', 2)
        self._max_features = config.get('num_words', MAX_NUM_WORDS)
        self.num_words = config.get('num_words', MAX_NUM_WORDS)
        self.num_classes = config.get('num_classes', 10)
        self.max_len = config.get('max_len', MAX_SEQUENCE_LENGTH)

    def add_ngram(self, sequences, token_indice, ngram_range=2):
        """
        Augment the input list of list (sequences) by appending n-grams values.
        Example: adding bi-gram
        >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
        >>> add_ngram(sequences, token_indice, ngram_range=2)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
        Example: adding tri-gram
        >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
        >>> add_ngram(sequences, token_indice, ngram_range=3)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
        """
        new_sequences = []
        for input_list in sequences:
            new_list = input_list[:]
            for ngram_value in range(2, ngram_range + 1):
                for i in range(len(new_list) - ngram_value + 1):
                    ngram = tuple(new_list[i:i + ngram_value])
                    if ngram in token_indice:
                        new_list.append(token_indice[ngram])
            new_sequences.append(new_list)

        return new_sequences

    def preprocess_input(self, X):
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

        tokens = [self.tokenize_fn(sent) for sent in X]
        tokens = self.tokenizer.texts_to_sequences(tokens)
        tokens = self.add_ngram(tokens, self.token_indice, self._ngrams)

        max_len = max([len(seq) for seq in tokens])
        if max_len > self.max_len:
            warnings.warn('Max training sequence length is %s, which is higher than max length setting %s' % \
                (max_len, self.max_len), UserWarning)

        tokens = pad_sequences(tokens, maxlen=self.max_len)

        return to_gpu(torch.LongTensor(tokens))

    def preprocess_output(self, y):
        # One-hot encode outputs
        # Can also use torch.eye() but leaving as numpy until torch achieves performance parity
        # lookup = np.eye(self.num_classes)
        # outputs = np.array([lookup[label] for label in y])
        # return to_gpu(torch.from_numpy(outputs).float())
        return to_gpu(torch.from_numpy(y).long())

    def get_layer_groups(self):
        model = self.model
        return [
            (model.embedding, model.emb_dropout),
            (model.i2h, model.h_dropout),
            model.h2o
        ]