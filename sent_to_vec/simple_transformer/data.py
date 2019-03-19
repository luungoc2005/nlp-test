import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from common.utils import word_to_vec, wordpunct_tokenize, pad_sequences
from sent_to_vec.masked_lm.data import read_wikitext
from config import START_TAG, STOP_TAG
import random

class TransformerSimpleLMDataset(Dataset):

    def initialize(self, data_path):
        if isinstance(data_path, str):
            self.raw_sents = read_wikitext(data_path)
            print('Loaded {} sentences from {}'.format(len(self.raw_sents), data_path))
        else:
            self.raw_sents = []
            for file_path in data_path:
                file_sents = read_wikitext(file_path)
                self.raw_sents.extend(file_sents)
                print('Loaded {} sentences from {}'.format(len(file_sents), file_path))
                print('Sample sentence: {}'.format(random.choice(file_sents)))

    def __len__(self):
        return len(self.raw_sents)

    def __getitem__(self, index):
        first_sent = self.raw_sents[index]
        
        return tokens, tags
            


def collate_transformer_entities_target(data):
    X_data = [item[0] for item in data]
    y_data = [item[1] for item in data]

    return X_data, torch.LongTensor(pad_sequences(y_data))