from keras.layers import multiply, Dense
from keras.layers.core import Lambda, RepeatVector, Permute, Reshape
from keras import backend as K

def apply_attention(inputs, single_attention_vector=True, name='attention_vec'):
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = int(inputs.shape[1])
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, time_steps))(a)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1), name = name + '_dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name=name)(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul