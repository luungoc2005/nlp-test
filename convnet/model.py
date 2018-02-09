from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, concatenate, multiply
from keras.layers.core import RepeatVector, Permute
from keras.models import Model
from keras import backend as K

from config import FILTER_SIZES, MAX_SEQUENCE_LENGTH, SENTENCE_DIM
from net_utils import apply_attention
from model_utils import fit_embedding_layers

def build_model(tokenizer, num_classes=10, non_static=True):
    input_layers = fit_embedding_layers(tokenizer)

    model = IntentConvNet( \
        tokens_input=input_layers['tokens_input'],
        pos_input=input_layers['pos_input'],
        static_embedding_layer=input_layers['static_embedding_layer'],
        non_static_embedding_layer=input_layers['non_static_embedding_layer'] if non_static else None,
        num_classes=num_classes)

    return model

def IntentConvNet(tokens_input=None,
                  pos_input=None,
                  static_embedding_layer=None, 
                  non_static_embedding_layer=None,
                  num_classes=10):
    # Allocate space for the 3 channels
    static_channels = FILTER_SIZES[:]
    non_static_channels = FILTER_SIZES[:]

    time_steps = int(static_embedding_layer.shape[1])
    input_dim = int(static_embedding_layer.shape[2])

    pos_attn = Dense(64)(pos_input) # single layer perceptron
    # pos_attn = Bidirectional(CuDNNLSTM(32))(pos_input)
    pos_attn = Dropout(0.5)(pos_attn)
    pos_attn = Dense(time_steps, activation='softmax')(pos_attn)
    pos_attn = RepeatVector(input_dim)(pos_attn)
    pos_attn = Permute((2, 1), name='pos_attention_vec')(pos_attn)

    static_attn_input = multiply([static_embedding_layer, pos_attn])
    non_static_attn_input = multiply([static_embedding_layer, pos_attn])

    for i, filter_size in enumerate(FILTER_SIZES):
        static_channels[i] = \
            Conv1D(MAX_SEQUENCE_LENGTH,
                   filter_size,
                   activation='relu',
                   padding='valid')(static_attn_input)
        static_channels[i] = \
            MaxPooling1D(MAX_SEQUENCE_LENGTH - filter_size + 1) \
            (static_channels[i])

        if non_static_embedding_layer is not None:
            non_static_channels[i] = \
                Conv1D(MAX_SEQUENCE_LENGTH,
                    filter_size,
                    activation='relu',
                    padding='valid')(non_static_attn_input)
            non_static_channels[i] = \
                MaxPooling1D(MAX_SEQUENCE_LENGTH - filter_size + 1) \
                (non_static_channels[i])

    static_conv = concatenate(static_channels)
    static_conv = Flatten()(static_conv)
    static_conv = Dropout(0.3)(static_conv)
    output_static = Dense(SENTENCE_DIM, activation='relu', name='static_output') \
                    (static_conv)

    if non_static_embedding_layer is not None:
        non_static_conv = concatenate(non_static_channels)
        non_static_conv = Flatten()(non_static_conv)
        non_static_conv = Dropout(0.3)(non_static_conv)
        output_non_static = Dense(SENTENCE_DIM, activation='relu', name='non_static_output') \
                            (non_static_conv)

    if non_static_embedding_layer is not None:
        x = concatenate([output_static, output_non_static])
    else:
        x = output_static
    
    main_output = Dense(num_classes, activation='softmax', name='main_output')(x)

    model = Model(inputs=[tokens_input, pos_input],
                  outputs=[main_output])
    model.summary()

    return model