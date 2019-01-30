from common.keras_preprocessing import Tokenizer
from common.wrappers import IModel
from sent_to_vec.sif.encoder import SIF_embedding
from common.utils import word_to_vec
from config import MAX_NUM_WORDS
from nltk.tokenize import wordpunct_tokenize
from featurizers.sif_featurizer import SIFFeaturizer
from sklearn.preprocessing import LabelEncoder
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import GridSearchCV
# import catboost as cb
from text_classification.utils.inference import infer_classification_output
import numpy as np
import torch


class EnsembleWrapper(IModel):

    def __init__(self, config={}, *args, **kwargs):
        super(EnsembleWrapper, self).__init__(
            # LogisticRegression,
            # class_weight='balanced',
            # multi_class='ovr',
            # tol=1e-4,
            # verbose=1
            # LightGBM - requires no class
            # object
            # SVC,
            # kernel='linear',
            # class_weight='balanced',
            # probability=True
            MLPClassifier,
            hidden_layer_sizes=(100,),
            activation='identity',
            verbose=1,
            featurizer=SIFFeaturizer(),
            # CatBoost
            # model_class=cb.CatBoostClassifier,
            # iterations=100,
            # loss_function='MultiClass',
            # eval_metric='Accuracy',
            # metric_period=10
            *args, **kwargs
        )
        self.config = config

        self.topk = config.get('top_k', 5)
        self.n_classes = config.get('num_classes', 10)

        # Distribution of params for GridSearchCV
        # self.param_dist = config.get('param_dist', 
        # {
        #     'depth': [4, 7, 10],
        #     'learning_rate' : [0.03, 0.1, 0.15],
        #     'l2_leaf_reg': [1,4,9],
        #     'iterations': [300]
        # }

        self.label_encoder = LabelEncoder()

    def get_state_dict(self):
        return {
            'config': self.config,
            'label_encoder': self.label_encoder,
            'state_dict': self.model
        }

    def load_state_dict(self, state_dict):
        # print(state_dict.keys())
        config = state_dict.get('config', dict())
        
        self.topk = config.get('top_k', 5)
        self.n_classes = config.get('num_classes', 10)

        # re-initialize model with loaded config
        # self.model = self.init_model()
        self.model = state_dict['state_dict']

        # load label encoder
        self.label_encoder = state_dict['label_encoder']
        
    def on_model_init(self):
        self._predict_fn = self.model.predict_proba

    def preprocess_output(self, y):
        # One-hot encode outputs
        # Can also use torch.eye() but leaving as numpy until torch achieves performance parity
        # lookup = np.eye(self.n_classes)
        # outputs = np.array([lookup[label] for label in y])
        # return torch.from_numpy(outputs).float()
        
        return torch.from_numpy(self.label_encoder.transform(y)).long()

    def infer_predict(self, logits, topk=None, context = []):
        logits = torch.from_numpy(logits).float()
        return infer_classification_output(self, logits, topk, context)