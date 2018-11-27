import torch
import numpy as np
from common.wrappers import IFeaturizer
from common.keras_preprocessing import Tokenizer
from common.utils import pad_sequences, word_to_vec
from nltk.tokenize import wordpunct_tokenize
from common.utils import n_letters
from config import LM_VOCAB_SIZE, LM_CHAR_RESERVED, START_TAG, STOP_TAG, UNK_TAG, MAX_SEQUENCE_LENGTH

class BasicFeaturizer(IFeaturizer):

    def __init__(self, config=dict()):
        super(BasicFeaturizer, self).__init__()

        self.lower = config.get('lower', True)
        self.char_level = config.get('char_level', False)
        self.num_words = config.get('num_words', n_letters + LM_CHAR_RESERVED if self.char_level else LM_VOCAB_SIZE)
        self.append_sos_eos = config.get('append_sos_eos', False)
        self.featurizer_seq_len = config.get('featurizer_seq_len', MAX_SEQUENCE_LENGTH)
        self.reserved_tokens = config.get('featurizer_reserved_tokens', [START_TAG, STOP_TAG, UNK_TAG])
        self.to_tensor = config.get('to_tensor', True) # pad sequences and return tensors

        self.tokenize_fn = wordpunct_tokenize
        
        self.tokenizer = Tokenizer(
            num_words=self.num_words, 
            lower=self.lower, 
            char_level=self.char_level,
            reserved_tokens=self.reserved_tokens
        )

    def get_output_shape(self):
        return (None,)

    def tokenize(self, data):
        if self.char_level:
            return data
        else:
            if self.append_sos_eos:
                return [[START_TAG] + self.tokenize_fn(sent) + [STOP_TAG] for sent in data]
            else:
                if isinstance(data[0], list):
                    return data
                else:
                    return [self.tokenize_fn(sent) for sent in data]

    def fit(self, data):
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(
                num_words=self.num_words, 
                lower=self.lower, 
                char_level=self.char_level,
                reserved_tokens=self.reserved_tokens
            )
        if self.char_level: print('Using char-level tokenizer')
        
        try:
            _ = (it for it in data)
            if len(data) < 1: return # Must have at least 1 item
        except:
            return # data is not an iterable
        
        self.tokenizer.fit_on_texts(self.tokenize(data))

    def transform(self, data, to_tensor=None):
        try:
            _ = (it for it in data)
            if len(data) < 1: return # Must have at least 1 item
        except:
            return # data is not an iterable

        tokens = self.tokenizer.texts_to_sequences(self.tokenize(data))

        if to_tensor if to_tensor is not None else self.to_tensor:
            lengths = [len(seq) for seq in tokens]
            max_seq_len = max(lengths)
            if self.featurizer_seq_len > 0:
                max_seq_len = min(max_seq_len, self.featurizer_seq_len)

            res = torch.zeros(len(tokens), max_seq_len).long()
            for idx, seq in enumerate(tokens):
                seq_len = min(max_seq_len, len(seq))
                res[idx, :seq_len] = torch.LongTensor(seq[:seq_len])

            return res
        else:
            return tokens

    def inverse_transform(self, data):
        retval = []
        batch_size = data.size(0)
        max_len = data.size(1)
        for ix in range(batch_size):
            retval.append(
                [
                    self.tokenizer.ix_to_word.get(int(data[ix, word_ix]), '')
                    for word_ix in range(max_len)
                ]
            )
        return retval