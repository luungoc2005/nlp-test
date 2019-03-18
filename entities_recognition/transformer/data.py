import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from common.utils import word_to_vec, wordpunct_space_tokenize, pad_sequences
from config import START_TAG, STOP_TAG

class TransformerEntitiesRecognitionDataset(Dataset):

    def __init__(self, training_data, tag_to_ix):
        super(TransformerEntitiesRecognitionDataset, self).__init__()

        self.tokenizer = wordpunct_space_tokenize
        self.samples = [item[0] for item in training_data]
        self.labels = [item[1] for item in training_data]
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.tokenizer(self.samples[index]), \
            [self.tag_to_ix[tag] for tag in self.labels[index].split()]


def collate_transformer_entities_target(data):
    X_data = [item[0] for item in data]
    y_data = [item[1] for item in data]

    return X_data, torch.LongTensor(pad_sequences(y_data))