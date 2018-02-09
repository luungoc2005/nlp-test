import pickle
import json

from os import path, getcwd
from tqdm import tqdm
from keras.models import load_model
from keras_tqdm import TQDMNotebookCallback
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from convnet.model import build_model
from model_utils import get_tokenizer_from_file, get_inputs, fit_embedding_layers
from net_utils import CyclicLR
from config import WORDS_PATH, BATCH_SIZE

BASE_PATH = path.join(getcwd(), 'convnet/')
LOG_DIR = path.join(BASE_PATH, 'logs/')
CHECKPOINT_PATH = path.join(BASE_PATH, 'model/weights-{epoch:02d}-{loss:.4f}.h5')
SAVE_PATH = path.join(BASE_PATH, 'model/model.h5')
TOKENIZER_PATH = path.join(BASE_PATH, 'model/tokenizer.pickle')
CLASSES_PATH = path.join(BASE_PATH, 'model/classes.json')

def train_model(X_train, 
                y_train, 
                classes=None, 
                tokenizer_path=None,
                model_path=None,
                load_weights_only=True,
                non_static=True,
                use_tqdm=True):
    if tokenizer_path is None:
        tokenizer = get_tokenizer_from_file(WORDS_PATH)
        tokenizer.fit_on_texts(X_train) # Add samples-unique vocabulary for the tokenizer

        with open(TOKENIZER_PATH, 'wb') as tokenizer_file:
            pickle.dump(tokenizer, tokenizer_file)
    else:
        with open(tokenizer_path, 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)

    num_classes = len(classes)
    with open(CLASSES_PATH, 'w') as classes_file:
        json.dump(classes, classes_file, indent=4)

    X_tokens, X_pos = get_inputs(X_train, tokenizer)

    # print(X_tokens.shape)
    # print(X_pos.shape)

    if model_path is None or load_weights_only:
        model = build_model(tokenizer, num_classes=num_classes, non_static=non_static)
    else:
        model = load_model(model_path, compile=False)

    model.compile(optimizer='adam', 
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    if not model_path is None and load_weights_only and path.isfile(model_path):
        model.load_weights(model_path, by_name=True)

    batch_size = min(BATCH_SIZE, len(X_train))

    # Workaround for a tqdm issue
    # https://github.com/tqdm/tqdm/issues/481
    tqdm.monitor_interval = 0
    callbacks = [
        CyclicLR(mode='triangular2'),
        TensorBoard(log_dir=LOG_DIR,
            write_graph=True,
            write_images=True, 
            write_grads=True,
            batch_size=batch_size),
        ModelCheckpoint(CHECKPOINT_PATH, 
            monitor='loss', 
            verbose=1, 
            save_best_only=True, 
            mode='min',
            period=2),
        EarlyStopping(monitor='loss', 
            min_delta=0.0001, 
            patience=5, 
            verbose=1, 
            mode='auto')
    ]

    if use_tqdm:
        callbacks.append(TQDMNotebookCallback())

    try:
        model.fit([X_tokens, X_pos], [y_train], 
                epochs=500, 
                batch_size=batch_size,
                callbacks=callbacks,
                shuffle=True,
                verbose=(0 if use_tqdm else 1))
    
        model.save(SAVE_PATH)
    except KeyboardInterrupt:
        model.save(SAVE_PATH)
