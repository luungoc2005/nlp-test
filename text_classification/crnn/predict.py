import torch

from config import SENTENCE_DIM
from text_classification.crnn.model import TextCRNN
from text_classification.crnn.train import SAVE_PATH
from common.utils import prepare_vec_sequence, word_to_vec, wordpunct_tokenize, topk


def load_model(save_path=None):
    save_path = save_path or SAVE_PATH
    data = torch.load(save_path)
    model = TextCRNN(classes=len(data['classes']))
    model.load_state_dict(data['state_dict'])
    return model, data['classes']


def predict(model, input_data, k=1):
    result = []
    for sentence in input_data:
        tokens_in = wordpunct_tokenize(sentence)
        sentence_in = prepare_vec_sequence([tokens_in], word_to_vec, SENTENCE_DIM, output='variable')
        scores = model(sentence_in)
        topk_scores = topk(scores, k)
        result.append(topk_scores)
    return result
