import torch
import numpy as np
from common.wrappers import IFeaturizer
from common.keras_preprocessing import Tokenizer
from common.utils import pad_sequences, word_to_vec
from nltk.tokenize import wordpunct_tokenize
from sent_to_vec.sif.encoder import SIF_embedding
from config import LM_VOCAB_SIZE

class BasicFeaturizer(IFeaturizer):

    def __init__(self, config=dict()):
        super(BasicFeaturizer, self).__init__()

        self.num_words = config.get('num_words', LM_VOCAB_SIZE)
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

        return torch.from_numpy(tokens).long()
