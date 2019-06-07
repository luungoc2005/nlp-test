from os import getcwd, path, listdir
import sys
from config import BASE_PATH

from common.utils import wordpunct_space_tokenize
from config import set_default_language

import io
import string

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default='pos')

args = parser.parse_args()

if __name__ == '__main__':
    set_default_language('en')

    TRAIN_PATH = path.join(BASE_PATH, 'data/CoNLL-2003/eng.train')
    print(TRAIN_PATH)
    
    def read_conll_2003(filename, tag_idx=-1):
        all_data = []

        current_txt = []
        current_tags = []
        tagset = []

        fin = io.open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
        fin_lines = fin.readlines()
        for line_ix, line in enumerate(fin_lines):
            line = line.strip()
            if len(line) > 0 and line_ix < len(fin_lines): # skip blank lines
                tmp = line.split('\t') if '\t' in line else line.split(' ')
                if tmp[0] != '-DOCSTART-':
                    current_txt.append(tmp[0])
                    current_tags.append(tmp[tag_idx])
                    tagset.append(tmp[tag_idx])
            else: # line is blank: finalize the token
                if len(current_txt) > 0:
                    line_txt = ''
                    line_tags = []
                    for idx in range(len(current_txt)):
                        tokens = wordpunct_space_tokenize(current_txt[idx])
                        if idx > 0:
                            line_txt += ' ' + current_txt[idx]
                            if current_tags[idx - 1] == current_tags[idx]:
                                line_tags.extend([current_tags[idx]] * (1 + len(tokens)))
                            else:
                                line_tags.append('-')
                                line_tags.extend([current_tags[idx]] * len(tokens))
                        else:
                            line_txt += current_txt[idx]
                            line_tags.extend([current_tags[idx]] * len(tokens))
                    all_data.append((line_txt, ' '.join(line_tags)))
                    current_txt = []
                    current_tags = []
        fin.close()

        tagset = list(set(tagset))

        return tagset, all_data

    dataset_tags, training_data = read_conll_2003(TRAIN_PATH, -1 if args.task == 'ner' else 1)
    tag_to_ix = {tag: key for key, tag in enumerate(list(set(dataset_tags)))}

    result = []
    for sentence, tag_seq in training_data:
        tokens_in = wordpunct_space_tokenize(sentence)
        assert len(tokens_in) == len(tag_seq.split(' '))

    print(training_data[30])
    print(tag_to_ix)

    from entities_recognition.transformer.model import TransformerSequenceTaggerWrapper
    from entities_recognition.transformer.train import TransformerSequenceTaggerLearner
    from entities_recognition.transformer.data import TransformerEntitiesRecognitionDataset
    from common.modules import BertAdam

    n_epochs = 50
    batch_size = 128
    model = TransformerSequenceTaggerWrapper({
        'tag_to_ix': tag_to_ix,
        'mode': 'lstm',
        'language': 'en'
    })
    learner = TransformerSequenceTaggerLearner(model, 
        optimizer_fn=BertAdam,
        optimizer_kwargs={
            'lr': 1e-3,
            'warmup': .1, 
            't_total': n_epochs * (len(training_data) // batch_size)
        })
    training_data = TransformerEntitiesRecognitionDataset(training_data, tag_to_ix)

    from common.callbacks import PrintLoggerCallback, EarlyStoppingCallback, ModelCheckpointCallback, TensorboardCallback, ReduceLROnPlateau
    learner.fit(
        training_data=training_data,
        epochs=n_epochs,
        batch_size=batch_size,
        callbacks=[
            PrintLoggerCallback(log_every=1),
            ReduceLROnPlateau(reduce_factor=4, patience=10),
            EarlyStoppingCallback(patience=50),
            ModelCheckpointCallback(prefix='en_conll_' if args.task == 'ner' else 'en_conll_pos', metrics=['loss'], every_epoch=5),
        ]
    )

    model.save('en-tagger.bin' if args.task == 'ner' else 'en-pos-tagger.bin')