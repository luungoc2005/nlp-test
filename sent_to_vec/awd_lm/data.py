import torch
from config import BASE_PATH, START_TAG, STOP_TAG, UNK_TAG, LM_SEQ_LEN
from nltk.tokenize import sent_tokenize
from torch.utils.data import Dataset
from os import path
from typing import Union, Iterable, Tuple

def read_wikitext(file_path):
    assert path.exists(file_path), '{} does not exist'.format(file_path)
    sents = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            if line.strip == '' or line.startswith(' = '):
                continue

            for sent in sent_tokenize(line):
                sents.append(sent \
                    .replace('<unk>', UNK_TAG) \
                    .replace('UNK', UNK_TAG)
                )

    return sents


class WikiTextDataset(Dataset):

    def __init__(self):
        super(WikiTextDataset, self).__init__()

    def initialize(self, model_wrapper, data_path, batch_size=64):
        if isinstance(data_path, str):
            self.raw_sents = read_wikitext(data_path)
            print('Loaded {} sentences from {}'.format(len(self.raw_sents), data_path))
        else:
            self.raw_sents = []
            for file_path in data_path:
                file_sents = read_wikitext(file_path)
                self.raw_sents.extend(file_sents)
                print('Loaded {} sentences from {}'.format(len(file_sents), file_path))

        self.seq_len = model_wrapper.config.get('seq_len', LM_SEQ_LEN)
        self.featurizer = model_wrapper.featurizer
        assert self.featurizer is not None

        print('Fitting featurizer')
        self.featurizer.fit(self.raw_sents)
        print('Found {} tokens'.format(len(self.featurizer.tokenizer.word_counts.keys())))

        print('Tokenizing files')
        raw_data = self.featurizer.transform(self.raw_sents)
        self.raw_data = raw_data
        # self.process_raw(batch_size)

    # def process_raw(self, batch_size):
    #     n_batch = self.raw_data.size(0) // batch_size
    #     batch_data = self.raw_data.narrow(0, 0, n_batch * batch_size)
    
    #     batch_data = batch_data.view(batch_size, -1).t().contiguous()
    #     self.batch_data = batch_data
    #     self.n_batch = n_batch

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
        # self.process_raw(batch_size)
        print('Finished loading preprocessed dataset')

    def __len__(self) -> int:
        return self.n_batch

    def __getitem__(self, index) -> Iterable:
        return self.raw_data[index]
        # seq_len = min(self.seq_len, len(self.batch_data) - 1 - index)
        # X = self.batch_data[index:index+seq_len].long()
        # y = self.batch_data[index+1:index+1+seq_len].view(-1)
        # return X, y

def collate_seq_fn(data, max_seq_len) -> Iterable:
    batch_data = torch.stack(data, 0) # (batch_size, seq_len)
        .t().contiguous() # (seq_len, batch_size)
    
    seq_len = min(max_seq_len, len(batch_data) - 1)
    X = batch_data[:seq_len].long()
    y = batch_data[1:1+seq_len].view(-1)

    return X, y
    

