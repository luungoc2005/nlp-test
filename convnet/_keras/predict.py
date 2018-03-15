from keras.models import load_model
from model_utils import get_inputs
from config import BATCH_SIZE
import pickle
import json

PREDICT_MODEL = None
TOKENIZER = None
CLASSES = None

def load_saved_model(model_path='', ignore_cache=False):
    global PREDICT_MODEL
    if PREDICT_MODEL is None or ignore_cache:
        PREDICT_MODEL = load_model(model_path)
    return PREDICT_MODEL

def load_tokenizer(tokenizer_path='', ignore_cache=False):
    global TOKENIZER
    if TOKENIZER is None or ignore_cache:
        with open(tokenizer_path, 'rb') as tokenizer_file:
            TOKENIZER = pickle.load(tokenizer_file)
    return TOKENIZER

def load_classes(classes_path='', ignore_cache=False):
    global CLASSES
    if CLASSES is None or ignore_cache:
        with open(classes_path, 'r') as classes_file:
            CLASSES = json.load(classes_file)
    return CLASSES

def run_predict(text, model=None, tokenizer=None):
    """
    Runs prediction with the cached model
    Accepts input as a list of text or a singular string
    """

    model = load_saved_model() if model is None else model
    tokenizer = load_tokenizer() if tokenizer is None else tokenizer

    if isinstance(text, list):
        input_data = text
    else:
        input_data = [text]

    batch_size = min(BATCH_SIZE, len(input_data))

    X_tokens, X_pos = get_inputs(input_data, tokenizer)

    result = model.predict([X_tokens, X_pos], batch_size=batch_size, verbose=0)

    return result

def interpret_netout(result, classes=None):
    """
    Generator for interpreting network outputs
    """

    classes = load_classes() if classes is None else classes

    for result_item in result:
        max_point = result_item.argmax()
        proba = result_item[max_point] * 100
        
        yield (classes[max_point], proba)
