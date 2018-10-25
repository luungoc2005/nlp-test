import torch
import numpy as np
from common.wrappers import IFeaturizer
from common.keras_preprocessing import Tokenizer
from common.utils import pad_sequences, word_to_vec
from nltk.tokenize import wordpunct_tokenize
from config import LM_VOCAB_SIZE, START_TAG, STOP_TAG, MAX_SEQUENCE_LENGTH

class BasicFeaturizer(IFeaturizer):

    def __init__(self, config=dict()):
        super(BasicFeaturizer, self).__init__()

        self.num_words = config.get('num_words', LM_VOCAB_SIZE)
        self.append_sos_eos = config.get('append_sos_eos', False)
        self.featurizer_seq_len = config.get('featurizer_seq_len', MAX_SEQUENCE_LENGTH)
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
        tokens = self.tokenizer.texts_to_sequences(self.tokenize(data))
        lengths = [len(seq) for seq in tokens]
        max_seq_len = max(lengths)
        if self.featurizer_seq_len > 0:
            max_seq_len = min(max_seq_len, self.featurizer_seq_len)

        res = torch.zeros(len(tokens), max_seq_len).long()
        for idx, seq in enumerate(tokens):
            seq_len = min(max_seq_len, len(seq))
            res[idx, :seq_len] = torch.LongTensor(seq[:seq_len])

        return res
