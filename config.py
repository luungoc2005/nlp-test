from os import path

CACHE_DATA = False

BASE_PATH = path.dirname(__file__)

GLOVE_PATH = path.join(BASE_PATH, 'data/fasttext/crawl-300d-2M.vec')
SKIP_FIRST_LINE = True # FastText format has n_words, n_dims as first line

MAGNITUDE_PATH = {
    'en': path.join(BASE_PATH, 'data/fasttext/crawl-300d-2M.magnitude')
}

VN_TREEBANK_PATH = path.join(BASE_PATH, 'data/vn_treebank')
# GLOVE_PATH = path.join(BASE_PATH, 'data/glove/glove.840B.300d.txt')
# SKIP_FIRST_LINE = False # FastText format has n_words, n_dims as first line

WORDS_PATH = path.join(BASE_PATH, 'data/20k.txt')

MAX_SEQUENCE_LENGTH = 70 # Max sentence length
MAX_NUM_WORDS = 50000 # Max vocabulary size
WORDS_SHORTLIST = 20000 # Common vocabulary size
EMBEDDING_DIM = 300 # GLOVE vector dimensions
SENTENCE_DIM = 50
# NGRAM_BINS = 2000000 # https://github.com/facebookresearch/fastText/blob/master/python/fastText/FastText.py
# NGRAM_BINS = 2 ** 21 # power of 2 bins that's close to 2m == 2097152
NGRAM_BINS = 2 ** 16 # == 65536

LM_VOCAB_SIZE = 200000 # Number of tokens in language model
LM_HIDDEN_DIM = 2048
LM_SEQ_LEN = 35

BATCH_SIZE = 32

# ConvNet configs (deprecated)
KERNEL_NUM = 100
FILTER_SIZES = [7, 8, 9]

# BiLSTM-CRF configs
START_TAG = "<START>"
STOP_TAG = "<STOP>"
UNK_TAG = "<UNK>"
EMPTY_TAG = "-"
EMBEDDING_DIM = 300
CHAR_EMBEDDING_DIM = 50
HIDDEN_DIM = 200
NUM_LAYERS = 1

# InferSent configs
SNLI_PATH = path.join(BASE_PATH, 'data/SNLI')
MultiNLI_PATH = path.join(BASE_PATH, 'data/MultiNLI')

QUORA_PATH = path.join(BASE_PATH, 'data/quora/quora_duplicate_questions.tsv')

# Flask configs
UPLOAD_FOLDER = path.join(BASE_PATH, 'flask_app/uploads')
LOGS_FOLDER = path.join(BASE_PATH, 'flask_app/logs')
CONFIG_PATH = path.join(BASE_PATH, 'flask_app/config.cfg')
PYTHON_PATH = 'python3'