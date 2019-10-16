import torch
import random
import re
import numpy as np
import math
import array
from torch.multiprocessing import Lock
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
    count = 0
    lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped == '' or stripped.startswith('=') or stripped.startswith('~~'):
                continue
            
            sent_batch = []
            for sent_line in sent_tokenize(line):
                processed_sent = sent_line \
                    .replace('<unk>', UNK_TAG) \
                    .replace('UNK', UNK_TAG)
                    # .replace('<UNK>', UNK_TAG) \

                for pattern in PATTERNS:
                    re.sub(pattern[0], pattern[1], processed_sent)

                sent_batch.append(processed_sent.strip())
            
            for ix, sent in enumerate(sent_batch):
                if ix == len(sent_batch) - 1:
                    count += 1
                    lines.append(sent.strip())
                else:
                    remaining_batch = sent_batch[ix:]
                    running_length = 0
                    running_sent = ''
                    for batch_ix, minibatch_sent in enumerate(remaining_batch):
                        if batch_ix == len(remaining_batch) - 1:
                            if len(running_sent) > 0:
                                count += 1
                                lines.append(running_sent.strip())
                            running_length = 0
                            running_sent = ''
                        elif running_length + len(minibatch_sent) > max_length:
                            if len(running_sent) > 0:
                                count += 1
                                lines.append(running_sent.strip())
                            running_length = 0
                            running_sent = ''
                            break
                        else:
                            running_sent += minibatch_sent if len(running_sent) == 0 else ' . ' + minibatch_sent
                            running_length += len(minibatch_sent) + 3

    with open(output_file, 'a+') as out_f:
        for line in lines:
            out_f.write(line + '\n')
    return count

class LanguageModelCorpusDataset(Dataset):

    def __init__(self):
        super(LanguageModelCorpusDataset, self).__init__()

    def init_on_corpus(self,
        data_path: Union[str, List[str]] = None, 
        data_texts: str = None,
        reset_path: bool = True,
        base_dir: str = BASE_PATH):

        if not hasattr(self, 'output_file') or not self.output_file:
            self.output_file = f'corpus-{uuid1()}.txt'
            self.sent_indices_file = f'index-{uuid1()}.txt'
            output_file = path.join(base_dir, self.output_file)
            sent_indices_file = path.join(base_dir, self.sent_indices_file)
        else:
            output_file = path.join(base_dir, self.output_file)
            sent_indices_file = path.join(base_dir, self.sent_indices_file)
        
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

        self.sent_indices = array.array('Q', [0] * line_count)
        current_idx = 0

        print('Caching sentence positions')
        with open(output_file, 'r') as output_fp:
            with open(sent_indices_file, 'w') as index_fp:
                for ix in tqdm(range(line_count)):
                    output_fp.readline()
                    
                    self.sent_indices[ix] = current_idx
                    index_fp.write(str(current_idx) + '\n')

                    current_idx = output_fp.tell()

        return line_count

    def init_on_model(
        self,
        model_wrapper: IModel, 
        data_path: Union[str, List[str]] = None, 
        data_texts: str = None,
        vocab_fp = None,
        base_dir: str = BASE_PATH
    ):
        self.line_count = self.init_on_corpus(data_path, data_texts, base_dir=base_dir)
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
        
        self._input_file = open(path.join(base_dir, self.output_file), 'r')
        self._read_lock = Lock()

    def save(self, save_path = 'maskedlm-data.bin'):
        torch.save({
            'featurizer': self.featurizer,
            'line_count': self.line_count,
            'output_file': self.output_file,
            'sent_indices': self.sent_indices_file
        }, save_path)
        print('Finished saving preprocessed dataset')

    def load(self, fp, model_wrapper, get_next_sent=False, base_dir: str = BASE_PATH):
        self.get_next_sent = get_next_sent
        state = torch.load(fp)
        self.featurizer = state['featurizer']
        model_wrapper.featurizer = state['featurizer']
        self.line_count = state['line_count']
        self.output_file = state['output_file']
        self.sent_indices_file = state.get('sent_indices', None)
        self.max_seq_len = model_wrapper.config.get('max_position_embeddings')

        if self.sent_indices_file is not None:
            with open(path.join(base_dir, self.sent_indices_file), 'r') as index_fp:
                self.sent_indices = array.array('Q', [0] * self.line_count)
                current_idx = 0
                while True:
                    line = index_fp.readline()
                    if not line:
                        break
                    else:
                        self.sent_indices[current_idx] = int(line)
                        current_idx += 1

        print(f'Finished loading preprocessed dataset. Corpus size: {len(self.sent_indices)}')
        self._input_file = open(path.join(base_dir, self.output_file), 'r')
        self._read_lock = Lock()

    def __len__(self) -> int:
        return self.line_count

    def get_sent(self, corpus_line: str) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_sent = self.featurizer.transform([
            corpus_line
        ])[0]

        if len(raw_sent) > self.max_seq_len:
            raw_sent = raw_sent[:self.max_seq_len]
        
        sent_length = raw_sent.size(0)
        output_label = torch.zeros(sent_length, dtype=torch.long)
        num_words = self.featurizer.tokenizer.num_words
        word_index = self.featurizer.tokenizer.word_index

        mask = torch.zeros(sent_length, dtype=torch.long)
        mask[:sent_length] = 1
        for ix in range(sent_length):
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
        return raw_sent, output_label, mask

    def __getitem__(self, index) -> Union[
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, bool]
        ]:
        try:
            self._read_lock.acquire()
            self._input_file.seek(self.sent_indices[index])
            
            if index == self.line_count - 1:
                text = self._input_file.read()
            else:
                text = self._input_file.read(self.sent_indices[index + 1] - self.sent_indices[index])

            self._read_lock.release()
            return self.get_sent(text.strip())
        except ValueError:
            print(f'Value error at {index}, file position: {self.sent_indices[index]}')
            raise ValueError
