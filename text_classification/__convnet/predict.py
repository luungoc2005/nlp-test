import torch
import torch.nn.functional as F
from config import SENTENCE_DIM
from text_classification.convnet.model import TextCNN
from text_classification.convnet.train import SAVE_PATH
from common.utils import prepare_vec_sequence, word_to_vec, wordpunct_tokenize


def load_model(save_path=None):
    save_path = save_path or SAVE_PATH
    data = torch.load(save_path)
    model = TextCNN(classes=len(data['classes']))
    model.load_state_dict(data['state_dict'])
    return model, data['classes']


def predict(model, input_data, k=1):
    with torch.no_grad():
        result = []
        for sentence in input_data:
            tokens_in = wordpunct_tokenize(sentence)
            sentence_in = prepare_vec_sequence(tokens_in, word_to_vec, SENTENCE_DIM, output='variable')
            scores = F.softmax(model(sentence_in), dim=-1)
            topk_scores = torch.topk(scores, k)

            result.append(topk_scores)
        return result
