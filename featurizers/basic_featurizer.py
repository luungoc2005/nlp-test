import torch
import numpy as np
from common.wrappers import IFeaturizer
from common.keras_preprocessing import Tokenizer
from common.utils import pad_sequences, word_to_vec
from nltk.tokenize import wordpunct_tokenize
from sent_to_vec.sif.encoder import SIF_embedding
from config import LM_VOCAB_SIZE, START_TAG, STOP_TAG

class BasicFeaturizer(IFeaturizer):

    def __init__(self, config=dict()):
        super(BasicFeaturizer, self).__init__()

        self.num_words = config.get('num_words', LM_VOCAB_SIZE)
        self.append_sos_eos = config.get('append_sos_eos', False)
        self.tokenize_fn = wordpunct_tokenize
        self.tokenizer = Tokenizer(num_words=self.num_words)

    def get_output_shape(self):
        return (None,)

    def tokenize(self, data):
        if self.append_sos_eos:
            return [[START_TAG] + self.tokenize_fn(sent) + [STOP_TAG] for sent in data]
        else:
            return [self.tokenize_fn(sent) for sent in data]

    def fit(self, data):
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        
        self.tokenizer.fit_on_texts(self.tokenize(data))

    def transform(self, data):
        tokens = np.array(self.tokenizer.texts_to_sequences(self.tokenize(data)))

        return torch.from_numpy(tokens).long()
