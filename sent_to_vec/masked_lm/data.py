import torch
import random
from config import BASE_PATH, START_TAG, STOP_TAG, UNK_TAG, EMPTY_TAG, MASK_TAG, LM_SEQ_LEN
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

    def initialize(self, model_wrapper, data_path):
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

        # self.seq_len = model_wrapper.config.get('seq_len', LM_SEQ_LEN)
        self.featurizer = model_wrapper.featurizer
        assert self.featurizer is not None

        print('Fitting featurizer')
        self.featurizer.fit(self.raw_sents)
        print('Found {} tokens'.format(len(self.featurizer.tokenizer.word_index.keys())))
        # print(list(self.featurizer.tokenizer.word_index.keys()))

        print('Tokenizing files')
        raw_data = self.featurizer.transform(self.raw_sents)
        self.raw_data = raw_data
        # self.process_raw(batch_size)

    def get_save_name(self):
        return 'wikitext-maskedlm-data.bin'

    def save(self):
        torch.save({
            'featurizer': self.featurizer,
            'data': self.raw_data,
            'raw_sents': self.raw_sents
        }, self.get_save_name())
        print('Finished saving preprocessed dataset')

    def load(self, fp, model_wrapper):
        state = torch.load(fp)
        self.featurizer = state['featurizer']
        model_wrapper.featurizer = state['featurizer']
        self.raw_data = state['data']
        # self.seq_len = model_wrapper.config.get('seq_len', LM_SEQ_LEN)
        # self.process_raw(batch_size)
        print('Finished loading preprocessed dataset')

    def __len__(self) -> int:
        # return self.n_batch
        return len(self.raw_data)

    def __getitem__(self, index) -> Iterable:
        # process sentence
        raw_sent = self.raw_data[index]
        output_label = torch.zeros(len(raw_sent)).long()
        for ix, token in enumerate(raw_sent):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
            else:
                output_label[ix] = self.featurizer.tokenizer.word_index[EMPTY_TAG]
        return 

def collate_seq_lm_fn(data, max_seq_len) -> Iterable:
    batch_data = torch.stack(data, 0) \
        .t().contiguous() # (seq_len, batch_size)
    
    seq_len = min(max_seq_len, len(batch_data) - 1)
    X = batch_data[:seq_len].long()
    y = batch_data[1:1+seq_len].view(-1)

    return X, y