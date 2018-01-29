import pickle

from os import path, getcwd
from tqdm import tqdm
from keras.models import load_model
from keras_tqdm import TQDMNotebookCallback
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from convnet.model import build_model
from model_utils import get_tokenizer_from_file, get_inputs, fit_embedding_layers
from config import WORDS_PATH

BASE_PATH = path.join(getcwd(), 'convnet/')
LOG_DIR = path.join(BASE_PATH, 'logs/')
CHECKPOINT_PATH = path.join(BASE_PATH, 'model/weights-{epoch:02d}-{loss:.4f}.h5')
SAVE_PATH = path.join(BASE_PATH, 'model/model.h5')
TOKENIZER_PATH = path.join(BASE_PATH, 'model/tokenizer.pickle')

def train_model(X_train, 
                y_train, 
                num_classes=10, 
                model_path=None, 
                use_tqdm=True):
    tokenizer = get_tokenizer_from_file(WORDS_PATH)

    with open(TOKENIZER_PATH, 'wb') as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file)

    X_tokens, X_pos = get_inputs(X_train, tokenizer)

    # print(X_tokens.shape)
    # print(X_pos.shape)

    if model_path is None:
        model = build_model(tokenizer, num_classes=num_classes)
    else:
        model = load_model(model_path)

    model.compile(optimizer='adam', 
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    batch_size = min(32, len(X_train))

    # Workaround for a tqdm issue
    # https://github.com/tqdm/tqdm/issues/481
    tqdm.monitor_interval = 0
    callbacks = [
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
