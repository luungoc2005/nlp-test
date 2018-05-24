import torch

from text_classification.fast_text.model import FastText
from text_classification.fast_text.train import SAVE_PATH, process_sentences


def load_model(save_path=None):
    save_path = save_path or SAVE_PATH
    data = torch.load(save_path)
    model = FastText(classes=len(data['classes']))
    model.load_state_dict(data['state_dict'])
    return model, data['classes']


def predict(model, input_data, k=1):
    result = []
    for sentence in input_data:
        tokens_in = process_sentences([sentence])
        scores = model(*tokens_in)
        topk_scores = torch.topk(scores, k)

        result.append(topk_scores)
    return result
