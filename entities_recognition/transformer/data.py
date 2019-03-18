import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from common.utils import word_to_vec, wordpunct_space_tokenize, pad_sequences
from config import START_TAG, STOP_TAG

class TransformerEntitiesRecognitionDataset(Dataset):

    def __init__(self, training_data, tag_to_ix, max_length=256):
        super(TransformerEntitiesRecognitionDataset, self).__init__()

        self.tokenizer = wordpunct_space_tokenize
        self.samples = [item[0] for item in training_data]
        self.labels = [item[1] for item in training_data]
        self.tag_to_ix = tag_to_ix
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        tokens = self.tokenizer(self.samples[index])
        tags = [self.tag_to_ix.get(tag, 0) for tag in self.labels[index].split()]
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            tags = tags[:self.max_length]
        return tokens, tags
            


def collate_transformer_entities_target(data):
    X_data = [item[0] for item in data]
    y_data = [item[1] for item in data]

    return X_data, torch.LongTensor(pad_sequences(y_data))