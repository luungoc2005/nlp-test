from os import getcwd, path, listdir
import sys
from config import BASE_PATH

from common.utils import wordpunct_space_tokenize
from config import set_default_language

import io
import string

if __name__ == '__main__':
    set_default_language('vi')

    TRAIN_PATH = path.join(BASE_PATH, 'data/vn_treebank')
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

    count = 0
    dataset_tags = []
    dataset_text = []
    for file_name in listdir(TRAIN_PATH):
        if file_name[-5:] == 'conll' and file_name[0] != '.':
            count += 1
            full_path = path.join(TRAIN_PATH, file_name)
            file_tags, file_data = read_conll_2003(full_path)
            print('Reading from %s, found %s items' % (full_path, len(file_data)))
            dataset_tags.extend(file_tags)
            dataset_text.extend(file_data)

    tag_to_ix = {tag: key + 1 for key, tag in enumerate(list(set(dataset_tags)))}

    print(tag_to_ix)

    for sentence, tag_seq in dataset_text:
        tokens_in = wordpunct_space_tokenize(sentence)
        assert len(tokens_in) == len(tag_seq.split(' '))
    #     print(read_tags(tokens_in, tag_seq.split(' ')))
    print('Loaded %s sentences from %s files' % (len(dataset_text), count))

    print(dataset_text[30])
    print(tag_to_ix)

    from entities_recognition.transformer.model import TransformerSequenceTaggerWrapper
    from entities_recognition.transformer.train import TransformerSequenceTaggerLearner
    from entities_recognition.transformer.data import TransformerEntitiesRecognitionDataset
    from common.modules import BertAdam

    n_epochs = 100
    batch_size = 128
    model = TransformerSequenceTaggerWrapper({'tag_to_ix': tag_to_ix})
    learner = TransformerSequenceTaggerLearner(model, 
        optimizer_fn=BertAdam,
        optimizer_kwargs={
            'lr': 1e-4
            # 'warmup': .1, 
            # 't_total': n_epochs * (len(dataset_text) // batch_size)
        })
    training_data = TransformerEntitiesRecognitionDataset(dataset_text, tag_to_ix)

    from common.callbacks import PrintLoggerCallback, EarlyStoppingCallback, ModelCheckpointCallback, TensorboardCallback, ReduceLROnPlateau
    learner.fit(
        training_data=training_data,
        epochs=n_epochs,
        batch_size=batch_size,
        callbacks=[
            PrintLoggerCallback(log_every=1),
            ReduceLROnPlateau(reduce_factor=4, patience=10),
            EarlyStoppingCallback(patience=50),
            ModelCheckpointCallback(prefix='vi_conll_', metrics=['loss'], every_epoch=5),
        ]
    )

    model.save('vn-tagger.bin')