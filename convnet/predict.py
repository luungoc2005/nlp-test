from keras.models import load_model
from model_utils import get_inputs
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

def run_predict(text, classes=None, model=None, tokenizer=None):
    model = load_saved_model() if model is None else model
    tokenizer = load_tokenizer() if tokenizer is None else tokenizer
    classes = load_classes() if classes is None else classes

    X_tokens, X_pos = get_inputs([text], tokenizer)

    result = model.predict([text], batch_size=1, verbose=0)

    max_point = result[0][0].argmax()

    proba = result[0][0][max_point] * 100

    print((classes[max_point], proba))

    return result