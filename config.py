from os import getcwd, path

BASE_PATH = getcwd()
GLOVE_PATH = path.join(BASE_PATH, 'data/glove/glove.6B.300d')
WORDS_PATH = path.join(BASE_PATH, 'data/20k.txt')

MAX_SEQUENCE_LENGTH = 512 # Max sentence length
MAX_NUM_WORDS = 20000 # Max vocabulary size
EMBEDDING_DIM = 300 # GLOVE vector dimensions
SENTENCE_DIM = 256

# ConvNet configs
FILTER_SIZES = [3, 4, 5]