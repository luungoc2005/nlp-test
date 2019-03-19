import torch
import numpy as np
from common.wrappers import IFeaturizer
from common.preprocessing.keras import Tokenizer
from common.utils import pad_sequences, word_to_vec
from common.word_vectors import most_similar
from nltk.tokenize import wordpunct_tokenize, word_tokenize
from sent_to_vec.sif.encoder import SIF_embedding
from config import MAX_NUM_WORDS

class SIFFeaturizer(IFeaturizer):

    def __init__(self, config=dict()):
        super(SIFFeaturizer, self).__init__()

        self.num_words = config.get('num_words', MAX_NUM_WORDS)
        self.tokenize_fn = word_tokenize
        self.use_tokenizer = config.get('use_tokenizer', False)

        self.tokenizer = Tokenizer(num_words=self.num_words)

    def get_output_shape(self):
        return (300,)

    def fit(self, data):
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        tokens = [self.tokenize_fn(sent) for sent in data]
        self.tokenizer.fit_on_texts(tokens)

    def transform(self, data):
        raw_tokens = [self.tokenize_fn(sent) for sent in data]
        tokens = self.tokenizer.texts_to_sequences(raw_tokens)
        tfidf_matrix = self.tokenizer.sequences_to_matrix(tokens, mode='tfidf')
        
        maxlen = max([len(sent) for sent in tokens])
        tfidf_weights = np.zeros((len(tokens), maxlen))
        for i, seq in enumerate(raw_tokens):
            for j, raw_token in enumerate(seq):
                token = -1
                if raw_token in self.tokenizer.word_index:
                    token = self.tokenizer.word_index[raw_token]
                else:
                    similar_to_raw_token = most_similar(raw_token)
                    for similar_word in similar_to_raw_token:
                        if similar_word in self.tokenizer.word_index:
                            token = self.tokenizer.word_index[raw_token]
                if token > -1:
                    tfidf_weights[i][j] = tfidf_matrix[i][token]
                else:
                    tfidf_weights[i][j] = 1 # default weight to 1
        
        # convert from token back to texts
        # this is to guarantee that tfidf matrix and X has the same length (with oov words ommited)
        # embs = word_to_vec(self.tokenizer.sequences_to_texts(tokens))
        embs = word_to_vec(raw_tokens)
        
        if embs is None: return None

        sif_emb = SIF_embedding(embs, tfidf_weights, rmpc=0)

        return torch.from_numpy(sif_emb).float()
