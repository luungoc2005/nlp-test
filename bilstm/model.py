from keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, concatenate

RECURRENT_DEPTH = 2
RECURRENT_UNITS = 32

def EntitiesBiLSTM():
    