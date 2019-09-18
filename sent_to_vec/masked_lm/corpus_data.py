import torch
import random
import re
import numpy as np
import math
# from common.torch_utils import to_gpu
from common.wrappers import IModel
from config import BASE_PATH, START_TAG, STOP_TAG, UNK_TAG, MASK_TAG, LM_SEQ_LEN
from torch.utils.data import Dataset
from nltk.tokenize import sent_tokenize
from os import path
from uuid import uuid1
from tqdm import tqdm

from typing import Union, Iterable, Tuple, List

PATTERNS = [
    (re.compile(r'[^\n]-[^\n]'), ' @-@ ')
]

def read_wikitext(input_file, output_file, max_length=128):
    assert path.exists(input_file), '{} does not exist'.format(input_file)
    out_f = open(output_file, 'a+')
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped == '' or stripped.startswith('=') or stripped.startswith('~~'):
                continue
            
            sent_batch = []
            for sent_line in sent_tokenize(line):
                processed_sent = sent_line \
                    .replace('<unk>', UNK_TAG) \
                    .replace('<UNK>', UNK_TAG) \
                    .replace('UNK', UNK_TAG)

                for pattern in PATTERNS:
                    re.sub(pattern[0], pattern[1], processed_sent)

                sent_batch.append(processed_sent.strip())
            
            for ix, sent in enumerate(sent_batch):
                if ix == len(sent_batch) - 1:
                    count += 1
                    out_f.write(sent.strip() + '\n')
                else:
                    remaining_batch = sent_batch[ix:]
                    running_length = 0
                    running_sent = ''
                    for batch_ix, minibatch_sent in enumerate(remaining_batch):
                        if batch_ix == len(remaining_batch) - 1:
                            if len(running_sent) > 0:
                                count += 1
                                out_f.write(running_sent.strip() + '\n')
                            running_length = 0
                            running_sent = ''
                        elif running_length + len(minibatch_sent) > max_length:
                            if len(running_sent) > 0:
                                count += 1
                                out_f.write(running_sent.strip() + '\n')
                            running_length = 0
                            running_sent = ''
                            break
                        else:
                            running_sent += minibatch_sent if len(running_sent) == 0 else ' . ' + minibatch_sent
                            running_length += len(minibatch_sent) + 3

    out_f.close()
    return count

class LanguageModelCorpusDataset(Dataset):

    def __init__(self):
        super(LanguageModelCorpusDataset, self).__init__()

    def init_on_corpus(self, 
        data_path: Union[str, List[str]] = None, 
        data_texts: str = None,
        reset_path: bool = True):

        if not hasattr(self, 'output_file') or not self.output_file:
            output_file = path.join(BASE_PATH, f'corpus-{uuid1()}.txt')
            self.output_file = output_file
        else:
            output_file = self.output_file

        line_count = 0

        if data_path is not None:
            if isinstance(data_path, str):
                data_path = [data_path]

            for file_path in data_path:
                file_sents = read_wikitext(file_path, output_file)
                line_count += file_sents
                print('Loaded {} sentences from {}'.format(file_sents, file_path))
        else:
            line_count = len(data_texts)

        self.sent_indices = np.zeros(line_count, dtype=np.int32)
        current_idx = 0

        print('Caching sentence positions')
        with open(self.output_file, 'r') as output_file:
            for ix in tqdm(range(line_count)):
                output_file.readline()
                self.sent_indices[ix] = current_idx
                current_idx = output_file.tell()

        return line_count

    def init_on_model(
        self,
        model_wrapper: IModel, 
        data_path: Union[str, List[str]] = None, 
        data_texts: str = None,
        vocab_fp = None,
    ):
        self.line_count = self.init_on_corpus(data_path, data_texts)
        self.max_seq_len = model_wrapper.config.get('max_position_embeddings')
        self.featurizer = model_wrapper.featurizer
        assert self.featurizer is not None

        if vocab_fp is not None:
            from common.preprocessing.keras import Tokenizer
            self.featurizer.tokenizer._load_from_json(vocab_fp)

        elif (len(self.featurizer.tokenizer.word_index) == 0):
            print('Fitting featurizer')
            if path.isfile(self.output_file):
                with open(self.output_file, 'r') as fp:
                    self.featurizer.fit(fp.readlines())
            else:
                FILE_BATCH_SIZE = 100000
                for batch_ix in tqdm(range(0, self.line_count, FILE_BATCH_SIZE)):
                    batch_path = path.join(self.output_file, str(batch_ix) + '.txt')
                    with open(batch_path, 'r') as fp:
                        self.featurizer.fit(fp.readlines())

        else:
            print('Featurizer previously fitted, continuing')

        print('Found {} tokens'.format(len(self.featurizer.tokenizer.word_index.keys())))

    def save(self, save_path = 'wikitext-maskedlm-data.bin'):
        torch.save({
            'featurizer': self.featurizer,
            'line_count': self.line_count,
            'output_file': self.output_file,
            'sent_indices': np.array(self.sent_indices, dtype=np.int32)
        }, save_path, pickle_protocol=4)
        print('Finished saving preprocessed dataset')

    def load(self, fp, model_wrapper, get_next_sent=False):
        self.get_next_sent = get_next_sent
        state = torch.load(fp)
        self.featurizer = state['featurizer']
        model_wrapper.featurizer = state['featurizer']
        self.line_count = np.array(state['line_count'], dtype=np.int32)
        self.output_file = state['output_file']
        self.sent_indices = state.get('sent_indices', [])
        self.max_seq_len = model_wrapper.config.get('max_position_embeddings')

        print(f'Finished loading preprocessed dataset. Corpus size: {self.line_count.shape}')

    def __len__(self) -> int:
        return self.line_count

    def get_sent(self, corpus_line: str) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_sent = self.featurizer.transform([
            corpus_line
        ])[0]

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

    def _get_raw_sent(self, index: int) -> str:
        with open(self.output_file) as input_file:
            input_file.seek(self.sent_indices[index])
            return input_file.readline().strip()

    def __getitem__(self, index) -> Union[
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, bool]
        ]:
        return self.get_sent(self._get_raw_sent(index))
