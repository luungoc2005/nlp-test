import torch

from text_classification.fast_text.model import FastText
from text_classification.fast_text.train import SAVE_PATH


def load_model(save_path=None):
    save_path = save_path or SAVE_PATH
    data = torch.load(save_path)
    model = FastText(classes=len(data['classes']))
    model.load_state_dict(data['state_dict'])
    model.detector = data['detector']
    return model, data['classes']


def predict(model, input_data, k=1):
    with torch.no_grad():
        result = []
        for sentence in input_data:
            scores = model(sentence, True)
            topk_scores, topk_idxs = torch.topk(scores, k)
            result.append(topk_scores)
        return result
