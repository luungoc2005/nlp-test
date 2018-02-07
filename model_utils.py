import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input
from keras.utils import to_categorical
from nltk.data import load
from nltk import pos_tag
from glove_utils import init_glove
from config import WORDS_PATH
from config import MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
from types import MethodType

# Get NLTK POS tags. Then append '#' (empty value) to the list
POS_TAGS = ['#'] + list(load('help/tagsets/upenn_tagset.pickle').keys())
POS_TAGS_COUNT = len(POS_TAGS)

def get_tokenizer(input_sequences, tokenizer_class=Tokenizer):
    tokenizer = tokenizer_class(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(input_sequences)
    tokenizer = add_word_map(tokenizer, force=True)

    return tokenizer

def add_word_map(tokenizer, force=False):
    if force or not hasattr(tokenizer, 'word_map'):
        tokenizer.word_map = dict(map(reversed, tokenizer.word_index.items()))
    return tokenizer

def get_tokenizer_from_file(input_file, tokenizer_class=Tokenizer):
    with open(WORDS_PATH, 'r') as words_file:
        content = words_file.readlines()
    tokenizer = tokenizer_class(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(content)

    return tokenizer

def get_inputs(input_sequences, tokenizer):
    token_sequences = tokenizer.texts_to_sequences(input_sequences)

    tokenizer = add_word_map(tokenizer)

    transformed_sequences = [
        ' '.join([tokenizer.word_map.get(word, '') for word in sequence])
        for sequence in token_sequences
    ]

    token_sequences = pad_sequences(token_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    pos_sequences = [
        np.array([
            POS_TAGS.index(tag)
            for token, tag in list(pos_tag(item))
        ], dtype='int32')
        for item in transformed_sequences
    ]
    pos_sequences = pad_sequences(pos_sequences,
                                  maxlen=MAX_SEQUENCE_LENGTH,
                                  value=POS_TAGS.index('#'))
    # Transform the sequence to categorical
    pos_sequences = [
        to_categorical(item, num_classes=POS_TAGS_COUNT)
        for item in pos_sequences
    ]
    pos_sequences = np.array(pos_sequences, dtype='float32')

    return token_sequences, pos_sequences

"""
Initializes Embedding layers with GLOVE weights
---
Returns:
An object with the following keys:
    'tokens_input'
    'pos_input'
    'static_embedding_layer'
    'non_static_embedding_layer'
"""
def fit_embedding_layers(tokenizer):
    embeddings_index = init_glove()

    tokens_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    pos_input = Input(shape=(MAX_SEQUENCE_LENGTH, POS_TAGS_COUNT,), dtype='float32')

    word_index = tokenizer.word_index
    word_index_len = len(word_index)
    print('Tokenizer contains %s unique tokens' % word_index_len)

    embedding_matrix = np.zeros((word_index_len + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros
            # TODO: replace with w2v for predicting unknown words
            embedding_matrix[i] = embedding_vector

    static_embedding_layer = \
        Embedding(word_index_len + 1,
                  EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_SEQUENCE_LENGTH,
                  trainable=False,
                  name='static_embeddings')(tokens_input)

    non_static_embedding_layer = \
        Embedding(word_index_len + 1,
                  EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_SEQUENCE_LENGTH,
                  trainable=True,
                  name='non_static_embeddings')(tokens_input)

    return {
        'tokens_input': tokens_input,
        'pos_input': pos_input,
        'static_embedding_layer': static_embedding_layer,
        'non_static_embedding_layer': non_static_embedding_layer,
    }
