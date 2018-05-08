import torch

from config import SENTENCE_DIM
from text_classification.convnet.model import TextCNN
from text_classification.convnet.train import SAVE_PATH
from common.utils import prepare_vec_sequence, word_to_vec, wordpunct_tokenize, topk


def load_model(num_classes):
    model = TextCNN(classes=num_classes)
    model.load_state_dict(torch.load(SAVE_PATH))
    return model


def predict(model, input_data, k=1):
    result = []
    for sentence in input_data:
        tokens_in = wordpunct_tokenize(sentence)
        sentence_in = prepare_vec_sequence(tokens_in, word_to_vec, SENTENCE_DIM, output='variable')
        scores = model(sentence_in)
        topk_scores = topk(scores, k)
        result.append(topk_scores)
    return result
