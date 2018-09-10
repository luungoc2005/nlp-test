from common.keras_preprocessing import Tokenizer
from common.wrappers import IModel
from common.torch_utils import to_gpu
from sent_to_vec.sif.encoder import SIF_embedding
from common.utils import word_to_vec
from config import MAX_NUM_WORDS, EMBEDDING_DIM
from nltk.tokenize import wordpunct_tokenize
from sklearn.preprocessing import LabelEncoder
from text_classification.utils.inference import infer_classification_output
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class MarginRankingLoss(nn.Module):
    def __init__(self, margin=.8, aggregate=torch.sum):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.aggregate = aggregate

    def forward(self, positive_similarity, negative_similarity):
        if positive_similarity.dim() == 1:
            positive_similarity = positive_similarity.unsqueeze(1)

        if negative_similarity.dim() == 1:
            negative_similarity = negative_similarity.unsqueeze(1)

        return self.aggregate(
            torch.clamp(self.margin - positive_similarity + negative_similarity, min=0))

class NegativeSampling():
    def __init__(self, n_output, n_negative=5, weights=None):
        super(NegativeSampling, self).__init__()
        self.n_output = n_output
        self.n_negative = n_negative
        self.weights = weights
        
    def sample(self, n_samples):
        if self.weights:
            samples = torch.multinomial(self.weights, n_samples, replacement=True)
        else:
            samples = torch.Tensor(n_samples).uniform_(0, self.n_output - 1).round().long()
        return samples

class InnerProductSimilarity(nn.Module):
    def __init__(self):
        super(InnerProductSimilarity, self).__init__()

    def forward(self, a, b):
        # a => B x [n_a x] dim, b => B x [n_b x] dim

        if a.dim() == 2:
            a = a.unsqueeze(1)  # B x n_a x dim

        if b.dim() == 2:
            b = b.unsqueeze(1)  # B x n_b x dim

        a = F.normalize(a, p=2, dim=-1)
        b = F.normalize(b, p=2, dim=-1)

        return torch.bmm(a, b.transpose(2, 1))  # B x n_a x n_b

class CosineSimilarity(nn.Module):
    def __init__(self):
        super(CosineSimilarity, self).__init__()
        self.similarity = nn.CosineSimilarity(dim=2)

    def forward(self, a, b):
        if a.dim() == 2:
            a = a.unsqueeze(1)  # B x n_a x dim

        if b.dim() == 2:
            b = b.unsqueeze(1)  # B x n_b x dim

        a = F.normalize(a, p=2, dim=-1)
        b = F.normalize(b, p=2, dim=-1)

        return self.similarity(a, b)

class StarSpaceClassifier(nn.Module):

    def __init__(self, config={}, *args, **kwargs):
        super(StarSpaceClassifier, self).__init__(*args, **kwargs)

        self.input_dim = config.get('input_dim', EMBEDDING_DIM)
        self.input_hidden_sizes = config.get('input_hidden_sizes', [300, 300])
        self.input_dropout_prob = config.get('input_dropout_prob', .2)
        self.output_hidden_sizes = config.get('output_hidden_sizes', [])
        self.output_dropout_prob = config.get('output_dropout_prob', .2)
        self.n_classes = config.get('num_classes', 10)
        self.max_norm = config.get('max_norm', 10)

        if len(self.input_hidden_sizes) > 0:
            input_emb_list = list()
            input_emb_list.append(nn.Linear(EMBEDDING_DIM, self.input_hidden_sizes[0]))
            if len(self.input_hidden_sizes) > 1:
                for idx, layer_size in enumerate(self.input_hidden_sizes[1:]):
                    input_emb_list.append(nn.Linear(self.input_hidden_sizes[idx], layer_size))
                    input_emb_list.append(nn.ReLU())
                    input_emb_list.append(nn.Dropout(self.input_dropout_prob))

            self.input_emb = nn.Sequential(*input_emb_list)
        else:
            self.input_emb = None

        if len(self.output_hidden_sizes) > 0:
            output_emb_list = list()
            output_emb_list.append(nn.Embedding(
                self.n_classes, 
                self.output_hidden_sizes[0], 
                max_norm=10.
            ))
            if len(self.output_hidden_sizes) > 1:
                for idx, layer_size in enumerate(self.output_hidden_sizes[1:]):
                    output_emb_list.append(nn.Linear(self.output_hidden_sizes[idx], layer_size))
                    output_emb_list.append(nn.ReLU())
                    output_emb_list.append(nn.Dropout(self.output_dropout_prob))
            self.output_emb = nn.Sequential(*output_emb_list)
        else:
            self.output_emb = nn.Embedding(
                self.n_classes, 
                self.input_hidden_sizes[-1] if len(self.input_hidden_sizes) > 0 else self.input_dim, 
                max_norm=10.
            )
        # self.similarity = InnerProductSimilarity()
        self.similarity = CosineSimilarity()
    
    def get_embs(self, input_embs=None, output=None):
        i_embs = None
        o_embs = None

        if input_embs is not None:
            if self.input_emb is not None:
                i_embs = self.input_emb(input_embs)
            else:
                i_embs = input_embs

        if output is not None:
            if output.ndimension() == 1:
                output = output.unsqueeze(-1)

            o_embs = self.output_emb(output)
            o_embs = torch.mean(o_embs, dim=1)
        
        return i_embs, o_embs

    def forward(self, input_embs):
        batch_size = input_embs.size(0)
        candidate_rhs = to_gpu(torch.arange(0, self.n_classes).long().expand(batch_size, -1))
        input_embs, candidate_rhs_repr = self.get_embs(
            input_embs, 
            candidate_rhs.contiguous().view(batch_size * self.n_classes)
        )
        candidate_rhs_repr = candidate_rhs_repr.view(batch_size, self.n_classes, -1)

        return self.similarity(input_embs, candidate_rhs_repr).squeeze(1)

class StarspaceClassifierWrapper(IModel):

    def __init__(self, config={}):
        super(StarspaceClassifierWrapper, self).__init__(
            model_class=StarSpaceClassifier, 
            config=config
        )

        self.tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        self.num_words = config.get('num_words', MAX_NUM_WORDS)
        self.n_classes = config.get('num_classes', 10)
        self.n_negative = config.get('n_negative', 20)
        self.loss_margin = config.get('loss_margin', .8)
        
        self.tokenize_fn = wordpunct_tokenize
        self.neg_sampling = NegativeSampling(
            n_output=self.n_classes, 
            n_negative=self.n_negative
        )
        self.label_encoder = LabelEncoder()

    def get_state_dict(self):
        return {
            'tokenizer': self.tokenizer,
            'config': self.model.config,
            'label_encoder': self.label_encoder,
            'state_dict': self.model.get_params(),
        }

    def load_state_dict(self, state_dict):
        config = state_dict['config']
        
        self.num_words = config.get('num_words', MAX_NUM_WORDS)
        self.n_classes = config.get('num_classes', 10)
        self.n_negative = config.get('n_negative', 20)
        self.loss_margin = config.get('loss_margin', .8)

        # re-initialize model with loaded config
        self.model = self.init_model()
        self.model.set_params(state_dict['state_dict'])

        # load tokenizer
        self.tokenizer = state_dict['tokenizer']

        # load label encoder
        self.label_encoder = state_dict['label_encoder']

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

        return torch.from_numpy(self.label_encoder.transform(y)).long()

    def infer_predict(self, logits, topk=None):
        return infer_classification_output(self, logits, topk)