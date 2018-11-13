import torch
import random
import numpy as np
from config import BASE_PATH, START_TAG, STOP_TAG, UNK_TAG, LM_SEQ_LEN
# from common.torch_utils import to_gpu
from nltk.tokenize import sent_tokenize
from torch.utils.data import Dataset
from os import path
from typing import Union, Iterable, Tuple

def read_wikitext_lm(file_path, char_level=False):
    assert path.exists(file_path), '{} does not exist'.format(file_path)
    sents = [] if not char_level else ''
    count = 0
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            if line.strip == '' or line.startswith(' = '):
                continue

            for sent in sent_tokenize(line):
                formatted_sent = sent \
                    .replace('<UNK>', UNK_TAG) \
                    .replace('<unk>', UNK_TAG) \
                    .replace('UNK', UNK_TAG)

                if not char_level:
                    formatted_sent = [START_TAG] + formatted_sent.split() + [STOP_TAG]

                    sents.extend(formatted_sent)
                else:
                    sents += formatted_sent
                
                count += 1

    return sents, count

class WikiTextDataset(Dataset):

    def __init__(self):
        super(WikiTextDataset, self).__init__()

    def initialize(self, model_wrapper, data_path, batch_size):
        char_level = model_wrapper.char_level

        if isinstance(data_path, str):
            self.raw_sents, sent_count = read_wikitext_lm(data_path, char_level=char_level)
            print('Loaded {} sentences from {}'.format(sent_count, data_path))
        else:
            self.raw_sents = []
            for file_path in data_path:
                file_sents, sent_count = read_wikitext_lm(file_path, char_level=char_level)
                self.raw_sents.extend(file_sents)
                print('Loaded {} sentences from {}'.format(sent_count, file_path))
                # print('Sample sentence: {}'.format(random.choice(file_sents)))

        self.seq_len = model_wrapper.config.get('seq_len', LM_SEQ_LEN)
        self.featurizer = model_wrapper.featurizer
        assert self.featurizer is not None

        if (len(self.featurizer.tokenizer.word_index) == 0):
            print('Fitting featurizer')
            self.featurizer.fit(self.raw_sents)
            # print(list(self.featurizer.tokenizer.word_index.keys()))
        else:
            print('Featurizer previously fitted, continuing')

        word_index = self.featurizer.tokenizer.word_index

        keys = list(word_index.keys())
        print('Found {} unique tokens'.format(len(keys)))
        print(keys[:10])

        print('Tokenizing files')
        if hasattr(self.featurizer, 'to_tensor'):
            self.featurizer.to_tensor = False
            raw_data = torch.LongTensor(
                self.featurizer.transform([self.raw_sents])
            )
            self.featurizer.to_tensor = True
        else:
            raw_data = self.featurizer.transform([self.raw_sents])

        self.raw_data = raw_data[0]
        self.process_raw(batch_size)

    def process_raw(self, batch_size=64):
        # print(self.raw_data.size())
        n_batch = self.raw_data.size(0) // batch_size
        print('Dataset contains {} minibatches with batch_size={}'.format(n_batch, batch_size))
        batch_data = self.raw_data.narrow(0, 0, n_batch * batch_size)
    
        batch_data = batch_data.view(batch_size, -1).t().contiguous()
        self.batch_data = batch_data
        # self.n_batch = n_batch
        self.n_batch = batch_data.size(0) - 1 - 1

    def get_save_name(self):
        return 'wikitext-data.bin'

    def save(self):
        torch.save({
            'featurizer': self.featurizer,
            'data': self.raw_data,
            'raw_sents': self.raw_sents
        }, self.get_save_name())
        print('Finished saving preprocessed dataset')

    def load(self, fp, model_wrapper, batch_size=64):
        state = torch.load(fp)
        self.featurizer = state['featurizer']
        model_wrapper.featurizer = state['featurizer']
        self.raw_data = state['data']
        self.seq_len = model_wrapper.config.get('seq_len', LM_SEQ_LEN)
        self.process_raw(batch_size)
        print('Finished loading preprocessed dataset')

    def __len__(self) -> int:
        return self.n_batch
        # return len(self.raw_data)

    def __getitem__(self, index) -> Iterable:
        # return self.raw_data[index]
        bptt = self.seq_len if np.random.random() < 0.95 else self.seq_len / 2
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        seq_len = min(seq_len, len(self.batch_data) - 1 - index)

        X = self.batch_data[index:index+seq_len].long()
        y = self.batch_data[index+1:index+1+seq_len].view(-1)

        return X, y
