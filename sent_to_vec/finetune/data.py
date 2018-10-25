from torch.utils.data import Dataset
from sent_to_vec.awd_lm.data import collate_seq_lm_fn
import random
import torch

class NextSentDataset(Dataset):

    def __init__(self):
        super(NextSentDataset, self).__init__()
        self.next_sent_threshold = 0.5

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

        self.featurizer = model_wrapper.featurizer
        assert self.featurizer is not None

        print('Fitting featurizer')
        self.featurizer.fit(self.raw_sents)
        print('Found {} tokens'.format(len(self.featurizer.tokenizer.word_counts.keys())))

        print('Tokenizing files')
        raw_data = self.featurizer.transform(self.raw_sents)
        self.raw_data = raw_data

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
        print('Finished loading preprocessed dataset')

    def __len__(self) -> int:
        return len(self.raw_data)

    def __getitem__(self, index) -> Iterable:
        if random.random() < self.next_sent_threshold:
            return self.raw_data[index], self.raw_data[index + 1], True
        else:
            return self.raw_data[index], random.choice(self.raw_data), False

def collate_seq_next_sent_fn(first_sents, second_sents, is_next, max_seq_len) -> Iterable:
    first_sents = collate_seq_lm_fn(first_sents, max_seq_len)
    second_sents = collate_seq_lm_fn(second_sents, max_seq_len)
    is_next = torch.Tensor(is_next).float()

    return first_sents, second_sents, is_next