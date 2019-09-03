import torch
import random
import re
# from common.torch_utils import to_gpu
from config import BASE_PATH, START_TAG, STOP_TAG, UNK_TAG, MASK_TAG, LM_SEQ_LEN
from nltk.tokenize import sent_tokenize
from torch.utils.data import Dataset
from os import path
from typing import Union, Iterable, Tuple
from tqdm import tqdm
import numpy as np

PATTERNS = [
    (re.compile(r'[^\n]-[^\n]'), ' @-@ ')
]
def read_wikitext(file_path):
    assert path.exists(file_path), '{} does not exist'.format(file_path)
    sents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped == '' or stripped.startswith('=') or stripped.startswith('~~'):
                continue

            processed_sent = line \
                .replace('<unk>', UNK_TAG) \
                .replace('<UNK>', UNK_TAG) \
                .replace('UNK', UNK_TAG)

            for pattern in PATTERNS:
                re.sub(pattern[0], pattern[1], processed_sent)

            sents.append(processed_sent)

    return sents


class WikiTextDataset(Dataset):

    def __init__(self):
        super(WikiTextDataset, self).__init__()

    def initialize(self, model_wrapper, data_path=None, data_texts=None, get_next_sent=False):
        self.get_next_sent = get_next_sent

        if data_path is not None:
            if isinstance(data_path, str):
                data_path = [data_path]

            self.raw_sents = []
            for file_path in data_path:
                file_sents = read_wikitext(file_path)
                self.raw_sents.extend(file_sents)
                print('Loaded {} sentences from {}'.format(len(file_sents), file_path))
                print('Sample sentence: {}'.format(random.choice(file_sents)))
        else:
            self.raw_sents = data_texts

        # self.seq_len = model_wrapper.config.get('seq_len', LM_SEQ_LEN)
        
        self.max_seq_len = model_wrapper.config.get('max_position_embeddings')
        self.featurizer = model_wrapper.featurizer
        assert self.featurizer is not None

        if (len(self.featurizer.tokenizer.word_index) == 0):
            print('Fitting featurizer')
            batch_size = 1024
            for i in tqdm(range(0, len(self.raw_sents), batch_size)):
                sent_batch = self.raw_sents[i:i+batch_size]
                self.featurizer.fit(sent_batch)

        else:
            print('Featurizer previously fitted, continuing')

        print('Found {} tokens'.format(len(self.featurizer.tokenizer.word_index.keys())))
        
        # print('Tokenizing files')
        # raw_data = self.featurizer.transform(self.raw_sents)
        # self.raw_data = raw_data
        # self.process_raw(batch_size)

    def get_save_name(self, num_words: int = None):
        if num_words is None:
            return 'wikitext-maskedlm-data.bin'
        else:
            return 'wikitext-maskedlm-data-{}.bin'.format(num_words)

    def save(self, save_path = ''):
        torch.save({
            'featurizer': self.featurizer,
            'raw_sents': self.raw_sents
        }, save_path if save_path != '' else self.get_save_name())
        print('Finished saving preprocessed dataset')

    def load(self, fp, model_wrapper, get_next_sent=False):
        self.get_next_sent = get_next_sent
        state = torch.load(fp)
        self.featurizer = state['featurizer']
        model_wrapper.featurizer = state['featurizer']
        self.raw_sents = np.array(state['raw_sents'], dtype=object)
        self.max_seq_len = model_wrapper.config.get('max_position_embeddings')

        # self.seq_len = model_wrapper.config.get('seq_len', LM_SEQ_LEN)
        # self.process_raw(batch_size)
        print('Finished loading preprocessed dataset')

    def __len__(self) -> int:
        # return self.n_batch
        return len(self.raw_sents)

    def get_sent(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        # process sentence
        raw_sent = self.featurizer.transform([self.raw_sents[index]])[0]

        if len(raw_sent) > self.max_seq_len:
            raw_sent = raw_sent[:self.max_seq_len]

        output_label = torch.LongTensor(len(raw_sent))
        num_words = self.featurizer.tokenizer.num_words
        word_index = self.featurizer.tokenizer.word_index

        for ix in range(raw_sent.size(0)):
            prob = random.random()
            if prob < 0.15:
                output_label[ix] = raw_sent[ix]

                prob /= 0.15
                if prob < 0.8:
                    raw_sent[ix] = word_index[MASK_TAG]
                
                elif prob < 0.9:
                    # 5 reserved tokens - hardcoded
                    raw_sent[ix] = random.randrange(4, num_words - 1)

                # else no change
            else:
                output_label[ix] = 0 # ignore idx
        return raw_sent, output_label

    def __getitem__(self, index) -> Union[
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, bool]
        ]:
        if not self.get_next_sent:
            return self.get_sent(index)
        else:
            first_sent = self.get_sent(index)
            if index == len(self.raw_sents) - 1 or random.random() < 0.5:
                return first_sent, self.get_sent(random.randrange(0, len(self.raw_sents) - 1)), False
            else:
                return first_sent, self.get_sent(index + 1), True

def collate_sent(data, max_seq_len):
    ret_val = torch.zeros(len(data), max_seq_len)
    for ix, seq in enumerate(data):
        seq_len = min(max_seq_len, len(seq))
        ret_val[ix, :seq_len] = seq[:seq_len]
    return ret_val.long().t().contiguous()

import math
def collate_sent_batch_first(data):
    max_seq_len = max([len(item) for item in data])
    max_seq_len = int(math.ceil(float(max_seq_len) / 8) * 8)
    ret_val = torch.zeros(len(data), max_seq_len)
    for ix, seq in enumerate(data):
        seq_len = min(max_seq_len, len(seq))
        ret_val[ix, :seq_len] = seq
    return ret_val.long().contiguous()

def collate_sent_target(data):
    X_data = [item[0] for item in data]
    y_data = [item[1] for item in data]

    max_len_X = max([len(item) for item in X_data])
    max_len_y = max([len(item) for item in y_data])
    max_len = max(max_len_X, max_len_y)
    max_len = int(math.floor(float(max_len) / 8) * 8)
    # return torch.stack(X_data, 0).long().t().contiguous(), torch.stack(y_data, 0).long().t().contiguous().view(-1)
    return collate_sent(X_data, max_len), collate_sent(y_data, max_len)

def collate_seq_lm_fn(data) -> Iterable:
    if len(data[0]) == 2: # first task
        return collate_sent_target(data)
    else: # second task
        first_batch, second_batch = collate_sent_target([(item[0], item[1]) for item in data])
        is_next = [item[2] for item in data]
        return first_batch, second_batch, torch.LongTensor(is_next)
    