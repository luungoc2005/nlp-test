import torch
import numpy as np
from common.wrappers import IFeaturizer
from common.utils import pad_sequences, word_to_vec
from common.word_vectors import get_magnitude_object
from nltk.tokenize import wordpunct_tokenize
from config import LM_VOCAB_SIZE, LM_CHAR_RESERVED, START_TAG, STOP_TAG, UNK_TAG, MAX_SEQUENCE_LENGTH

class ElmoFeaturizer(IFeaturizer):

    def __init__(self, config=dict()):
        super(ElmoFeaturizer, self).__init__()

        self.lower = config.get('lower', True)
        self.featurizer_seq_len = config.get('featurizer_seq_len', MAX_SEQUENCE_LENGTH)
        self.reserved_tokens = config.get('featurizer_reserved_tokens', [START_TAG, STOP_TAG, UNK_TAG])
        self.to_tensor = config.get('to_tensor', True) # pad sequences and return tensors
        self.return_mask = config.get('return_mask', False)
        self.concat = config.get('concat', 'mean')

        self.tokenize_fn = wordpunct_tokenize

    def get_output_shape(self):
        return (None,)

    def tokenize(self, data):
        if isinstance(data[0], list):
            return data
        else:
            return [self.tokenize_fn(sent) for sent in data]

    def fit(self, data):
        pass

    def transform(self, data, return_mask=None):
        tensor = word_to_vec(data, language='en_elmo')
        for sent in tensor:
            print(np.array(sent, dtype='float32').shape)
        tensor = np.array(tensor, dtype='float32')
        print(tensor.shape)
        if self.concat == 'mean':
            tensor = np.mean(tensor, axis=1)
        elif self.concat == 'max':
            tensor = np.max(tensor, axis=1)
        return tensor

    def inverse_transform(self, data):
        return data