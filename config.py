from os import path

CACHE_DATA = False

BASE_PATH = path.dirname(__file__)

GLOVE_PATH = path.join(BASE_PATH, 'data/fasttext/crawl-300d-2M.vec')
SKIP_FIRST_LINE = True # FastText format has n_words, n_dims as first line

# GLOVE_PATH = path.join(BASE_PATH, 'data/glove/glove.840B.300d.txt')
# SKIP_FIRST_LINE = False # FastText format has n_words, n_dims as first line

WORDS_PATH = path.join(BASE_PATH, 'data/20k.txt')

MAX_SEQUENCE_LENGTH = 256 # Max sentence length
MAX_NUM_WORDS = 50000 # Max vocabulary size
WORDS_SHORTLIST = 20000 # Common vocabulary size
EMBEDDING_DIM = 300 # GLOVE vector dimensions
SENTENCE_DIM = 50
NGRAM_BINS = 2000000 # https://github.com/facebookresearch/fastText/blob/master/python/fastText/FastText.py

BATCH_SIZE = 32

# ConvNet configs (deprecated)
KERNEL_NUM = 100
FILTER_SIZES = [7, 8, 9]

# BiLSTM-CRF configs
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 300
CHAR_EMBEDDING_DIM = 50
HIDDEN_DIM = 200
NUM_LAYERS = 1

# InferSent configs
NLI_PATH = path.join(BASE_PATH, 'data/SNLI')

QUORA_PATH = path.join(BASE_PATH, 'data/quora/quora_duplicate_questions.tsv')

# Flask configs
UPLOAD_FOLDER = 'flask_app/uploads'