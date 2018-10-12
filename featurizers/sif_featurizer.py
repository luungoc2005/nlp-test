import torch
import numpy as np
from common.wrappers import IFeaturizer
from common.keras_preprocessing import Tokenizer
from common.utils import pad_sequences, word_to_vec
from nltk.tokenize import wordpunct_tokenize
from sent_to_vec.sif.encoder import SIF_embedding
from config import MAX_NUM_WORDS

class SIFFeaturizer(IFeaturizer):

    def __init__(self, config=dict()):
        super(SIFFeaturizer, self).__init__()

        self.num_words = config.get('num_words', MAX_NUM_WORDS)
        self.tokenize_fn = wordpunct_tokenize
        self.tokenizer = Tokenizer(num_words=self.num_words)

    def get_output_shape(self):
        return (300,)

    def fit(self, data):
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        tokens = [self.tokenize_fn(sent) for sent in data]
        self.tokenizer.fit_on_texts(tokens)

    def transform(self, data):
        tokens = [self.tokenize_fn(sent) for sent in data]
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
