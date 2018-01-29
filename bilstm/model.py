from keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, concatenate
from keras.layers.wrappers import TimeDistributed
from keras.models import Model

RECURRENT_DEPTH = 2
RECURRENT_UNITS = 32

def EntitiesBiLSTM(num_classes):
    input_tokens, input_pos
    tokens_lstm_layers = list(range(RECURRENT_DEPTH))
    pos_lstm_layers = list(range(RECURRENT_DEPTH))

    for idx in range(RECURRENT_DEPTH):
        tokens_lstm_layers[idx] = Dropout(0.3)(input_tokens if idx == 0 else tokens_lstm_layers[idx-1])
        tokens_lstm_layers[idx] = Bidirectional(LSTM(RECURRENT_UNITS, return_sequences=True))(tokens_lstm_layers[idx])
        tokens_lstm_layers[idx] = Dropout(0.3)(tokens_lstm_layers[idx])

        pos_lstm_layers[idx] = Dropout(0.3)(input_pos if idx == 0 else pos_lstm_layers[idx-1])
        pos_lstm_layers[idx] = Bidirectional(LSTM(RECURRENT_UNITS, return_sequences=True))(pos_lstm_layers[idx])
        pos_lstm_layers[idx] = Dropout(0.3)(pos_lstm_layers[idx])

    x = concatenate([tokens_lstm_layers[-1], pos_lstm_layers[-1]])
    output = TimeDistributed(Dense(num_classes, activation='softmax'))(x)

    model = Model(inputs=[input_tokens, input_pos], outputs=[output])

    return model