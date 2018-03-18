from os import getcwd, path

BASE_PATH = getcwd()
GLOVE_PATH = path.join(BASE_PATH, 'data/glove/glove.6B.300d')
WORDS_PATH = path.join(BASE_PATH, 'data/20k.txt')
FASTTEXT_BIN = path.join(BASE_PATH, 'data/fasttext/wiki.en.bin')
FASTTEXT_VEC = path.join(BASE_PATH, 'data/fasttext/wiki-news-300d-1M-subword')

MAX_SEQUENCE_LENGTH = 50 # Max sentence length
# MAX_NUM_WORDS = 21000 # Max vocabulary size (plus some extra vocab by the training examples)
EMBEDDING_DIM = 300 # GLOVE vector dimensions
SENTENCE_DIM = 20

BATCH_SIZE = 32

# ConvNet configs (deprecated)
KERNEL_NUM = 100
FILTER_SIZES = [7, 8, 9]

# BiLSTM-CRF configs
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 300
HIDDEN_DIM = 512