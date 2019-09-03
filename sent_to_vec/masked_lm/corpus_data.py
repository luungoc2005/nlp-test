import torch
import random
import re
import numpy as np
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
def read_wikitext(input_file, output_file):
    assert path.exists(input_file), '{} does not exist'.format(input_file)
    sents = []
    out_f = open(output_file, 'a+')
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        count += 1
        for line in f:
            stripped = line.strip()
            if stripped == '' or stripped.startswith('=') or stripped.startswith('~~'):
                continue

            for sent in sent_tokenize(line):
                processed_sent = sent \
                    .replace('<unk>', UNK_TAG) \
                    .replace('<UNK>', UNK_TAG) \
                    .replace('UNK', UNK_TAG)

                for pattern in PATTERNS:
                    re.sub(pattern[0], pattern[1], processed_sent)

                out_f.write(processed_sent.strip() + '\n')
    out_f.close()
    return count

class LanguageModelCorpusDataset(Dataset):

    def __init__(self):
        super(LanguageModelCorpusDataset, self).__init__()

    def init_on_model(
        self,
        model_wrapper: IModel, 
        data_path: Union[str, List[str]] = None, 
        data_texts: str = None
    ):
        output_file = path.join(BASE_PATH, f'corpus-{uuid1()}.txt')
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
        
        self.max_seq_len = model_wrapper.config.get('max_position_embeddings')
        self.featurizer = model_wrapper.featurizer
        assert self.featurizer is not None

        if (len(self.featurizer.tokenizer.word_index) == 0):
            print('Fitting featurizer')
            batch_size = 1024
            for i in tqdm(range(0, line_count, batch_size)):
                sent_batch = []
                
                # sent_batch = self.raw_sents[i:i+batch_size]
                
                self.featurizer.fit(sent_batch)
