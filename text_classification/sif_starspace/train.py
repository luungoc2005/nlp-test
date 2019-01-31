import torch
import numpy as np
import torch.nn as nn
from common.torch_utils import to_gpu
from common.wrappers import ILearner
from common.metrics import accuracy, recall, precision, f1
from text_classification.sif_starspace.model import MarginRankingLoss, NegativeSampling
from sklearn.utils import class_weight
from common.utils import to_categorical
from config import EMBEDDING_DIM
import lightgbm as lgb

class StarspaceClassifierLearner(ILearner):

    def __init__(self, model, n_negative=20, *args, **kwargs):
        super(StarspaceClassifierLearner, self).__init__(model, preprocess_batch=False, *args, **kwargs)
        self.n_negative = n_negative

    def init_on_data(self, X, y):
        tokens = [self.model_wrapper.tokenize_fn(sent) for sent in X]
        self.model_wrapper.tokenizer.fit_on_texts(tokens)
        self.n_samples = len(tokens)
        # self.n_classes = len(np.unique(y))
        self.buffer_pointer = 0
        

        self.model_wrapper.label_encoder.fit(y)

        n_classes = len(self.model_wrapper.label_encoder.classes_)
        self.model_wrapper.config['num_classes'] = self.model_wrapper.n_classes

        self.neg_sampling = to_gpu(NegativeSampling(
            n_output=n_classes, 
            n_negative=self.n_negative
        ))
        # self.class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)

        config = self.model_wrapper.config
        if 'contexts' in config:
            contexts = config['contexts']
            contexts_list = [
                contexts[label] if label in contexts else []
                for label in self.model_wrapper.label_encoder.classes_
            ]
            self.model_wrapper.config['contexts'] = contexts_list
            # print('number of contexts: %s' % str(len(contexts_list)))

    def on_model_init(self):
        self.criterion = to_gpu(MarginRankingLoss(margin=self.model_wrapper.loss_margin))
        # self.criterion = nn.CosineEmbeddingLoss(margin=.8, reduction='sum')

    def on_epoch(self, X, y):
        model = self.model_wrapper.model

        input_embs, pos_output_embs = model.get_embs(X, y)
        positive_similarity = model.similarity(input_embs, pos_output_embs).squeeze(1)
        # print(positive_similarity.size())

        batch_size = X.size(0)
        n_samples = batch_size * self.n_negative
        neg_rhs = to_gpu(self.neg_sampling.sample(n_samples))
        
        _, neg_output_embs = model.get_embs(output=neg_rhs)  # (B * n_negative) x dim
        neg_output_embs = neg_output_embs.view(batch_size, self.n_negative, -1)  # B x n_negative x dim
        negative_similarity = model.similarity(input_embs, neg_output_embs).squeeze(1)  # B x n_negative
        # print(negative_similarity.size())

        similarity = model(X)

        loss = self.criterion(positive_similarity, negative_similarity)
        loss.backward()

        return {
            'loss': loss.detach().item(), 
            'logits': torch.max(similarity, dim=-1)[1]
        }

    def calculate_metrics(self, logits, y):
        return {
            'accuracy': accuracy(logits, y)
            # 'f1': f1(logits, y),
            # 'precision': precision(logits, y),
            # 'recall': recall(logits, y)
        }