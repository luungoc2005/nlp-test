from common.keras_preprocessing import Tokenizer
from common.wrappers import IModel
from sent_to_vec.sif.encoder import SIF_embedding
from common.utils import word_to_vec
from config import MAX_NUM_WORDS
from nltk.tokenize import wordpunct_tokenize
from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import GridSearchCV
# import catboost as cb
import numpy as np
import torch


class EnsembleWrapper(IModel):

    def __init__(self, config):
        super(EnsembleWrapper, self).__init__(
            # LogisticRegression,
            # class_weight='balanced',
            # multi_class='ovr',
            # tol=1e-4,
            # verbose=1
            # LightGBM - requires no class
            # object
            SVC,
            kernel='linear',
            class_weight='balanced',
            probability=True
            # MLPClassifier,
            # hidden_layer_sizes=(100,),
            # activation='identity',
            # verbose=1
            # CatBoost
            # model_class=cb.CatBoostClassifier,
            # iterations=100,
            # loss_function='MultiClass',
            # eval_metric='Accuracy',
            # metric_period=10
        )

        self.tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        self.num_words = config.get('num_words', MAX_NUM_WORDS)
        self.num_classes = config.get('num_classes', 10)

        # Distribution of params for GridSearchCV
        # self.param_dist = config.get('param_dist', 
        # {
        #     'depth': [4, 7, 10],
        #     'learning_rate' : [0.03, 0.1, 0.15],
        #     'l2_leaf_reg': [1,4,9],
        #     'iterations': [300]
        # }
        
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

    def infer_predict(self, logits): return torch.from_numpy(logits).float()
