from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, concatenate
from keras.models import Model

from config import FILTER_SIZES, MAX_SEQUENCE_LENGTH, SENTENCE_DIM
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
    pos_channels = FILTER_SIZES[:]

    for i, filter_size in enumerate(FILTER_SIZES):
        static_channels[i] = \
            Conv1D(MAX_SEQUENCE_LENGTH,
                   filter_size,
                   activation='relu',
                   padding='valid')(static_embedding_layer)
        static_channels[i] = \
            MaxPooling1D(MAX_SEQUENCE_LENGTH - filter_size + 1) \
            (static_channels[i])

        if non_static_embedding_layer is not None:
            non_static_channels[i] = \
                Conv1D(MAX_SEQUENCE_LENGTH,
                    filter_size,
                    activation='relu',
                    padding='valid')(non_static_embedding_layer)
            non_static_channels[i] = \
                MaxPooling1D(MAX_SEQUENCE_LENGTH - filter_size + 1) \
                (non_static_channels[i])

        pos_channels[i] = \
            Conv1D(MAX_SEQUENCE_LENGTH,
                   filter_size,
                   activation='relu',
                   padding='valid')(pos_input)
        pos_channels[i] = \
            MaxPooling1D(MAX_SEQUENCE_LENGTH - filter_size + 1) \
            (pos_channels[i])

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

    pos_conv = concatenate(pos_channels)
    pos_conv = Flatten()(pos_conv)
    pos_conv = Dropout(0.3)(pos_conv)
    output_pos = Dense(SENTENCE_DIM, activation='relu', name='pos_output') \
                 (pos_conv)

    if non_static_embedding_layer is not None:
        x = concatenate([output_static, output_non_static, output_pos])
    else:
        x = concatenate([output_static, output_pos])
    
    main_output = Dense(num_classes, activation='softmax', name='main_output')(x)

    model = Model(inputs=[tokens_input, pos_input],
                  outputs=[main_output])
    model.summary()

    return model