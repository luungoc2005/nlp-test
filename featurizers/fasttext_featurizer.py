import torch
import numpy as np
from common.wrappers import IFeaturizer
from common.preprocessing.keras import Tokenizer
from common.utils import pad_sequences, word_to_vec
from common.torch_utils import to_gpu
from nltk.tokenize import wordpunct_tokenize
from sent_to_vec.sif.encoder import SIF_embedding
from config import MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH

class FastTextFeaturizer(IFeaturizer):

    def __init__(self, config=dict()):
        super(FastTextFeaturizer, self).__init__()

        self.num_words = config.get('num_words', MAX_NUM_WORDS)
        self.tokenize_fn = wordpunct_tokenize
        self.tokenizer = Tokenizer(num_words=self.num_words)
        self.token_indice = None
        self.indice_token = None
        self.max_features = MAX_NUM_WORDS
        self.max_len = MAX_SEQUENCE_LENGTH
        self.ngrams = 3

    def get_output_shape(self):
        return (300,) # size of embedding

    def create_ngram_set(self, input_list, ngram_value=2):
        """
        Extract a set of n-grams from a list of integers.
        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
        {(4, 9), (4, 1), (1, 4), (9, 4)}
        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
        [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
        """
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))

    def add_ngram(self, sequences, token_indice, ngram_range=2):
        """
        Augment the input list of list (sequences) by appending n-grams values.
        Example: adding bi-gram
        >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
        >>> add_ngram(sequences, token_indice, ngram_range=2)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
        Example: adding tri-gram
        >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
        >>> add_ngram(sequences, token_indice, ngram_range=3)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
        """
        new_sequences = []
        for input_list in sequences:
            new_list = input_list[:]
            for ngram_value in range(2, ngram_range + 1):
                for i in range(len(new_list) - ngram_value + 1):
                    ngram = tuple(new_list[i:i + ngram_value])
                    if ngram in token_indice:
                        new_list.append(token_indice[ngram])
            new_sequences.append(new_list)

        return new_sequences

    def fit(self, data):
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        tokens = [self.tokenize_fn(sent) for sent in data]
        self.tokenizer.fit_on_texts(tokens)

        if self.ngrams > 1:
            # print('Adding {}-gram features'.format(ngram_range))
            # Create set of unique n-gram from the training set.
            ngram_set = set()
            for input_list in data:
                for i in range(2, self.ngrams + 1):
                    set_of_ngram = self.create_ngram_set(input_list, ngram_value=i)
                    ngram_set.update(set_of_ngram)

        start_index = self.num_words + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}
        self.token_indice = token_indice
        self.indice_token = indice_token

        max_features = np.max(list(indice_token.keys())) + 1
        self.max_features = max_features

    def transform(self, data):
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

        tokens = [self.tokenize_fn(sent) for sent in data]
        tokens = self.tokenizer.texts_to_sequences(tokens)
        tokens = self.add_ngram(tokens, self.token_indice, self.ngrams)

        max_len = max([len(seq) for seq in tokens])
        if max_len > self.max_len:
            warnings.warn('Max training sequence length is %s, which is higher than max length setting %s' % \
                (max_len, self.max_len), UserWarning)

        tokens = pad_sequences(tokens, maxlen=self.max_len)

        return to_gpu(torch.LongTensor(tokens))
