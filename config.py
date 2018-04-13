from os import path

CACHE_DATA = False

BASE_PATH = path.dirname(__file__)
GLOVE_PATH = path.join(BASE_PATH, 'data/glove/glove.6B.300d')
WORDS_PATH = path.join(BASE_PATH, 'data/20k.txt')
FASTTEXT_BIN = path.join(BASE_PATH, 'data/fasttext/wiki.en.bin')
FASTTEXT_VEC = path.join(BASE_PATH, 'data/fasttext/wiki-news-300d-1M-subword')

MAX_SEQUENCE_LENGTH = 50 # Max sentence length
MAX_NUM_WORDS = 500000 # Max vocabulary size
EMBEDDING_DIM = 300 # GLOVE vector dimensions
SENTENCE_DIM = 50
NGRAM_BINS = 20000 # 2000000 # https://github.com/facebookresearch/fastText/blob/master/python/fastText/FastText.py

BATCH_SIZE = 32

# ConvNet configs (deprecated)
KERNEL_NUM = 100
FILTER_SIZES = [7, 8, 9]

# BiLSTM-CRF configs
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 300
CHAR_EMBEDDING_DIM = 50
HIDDEN_DIM = 100
NUM_LAYERS = 2

# InferSent configs
NLI_PATH = path.join(BASE_PATH, 'data/SNLI')