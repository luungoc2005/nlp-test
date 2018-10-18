from config import BASE_PATH, START_TAG, STOP_TAG, UNK_TAG
from nltk.tokenize import sent_tokenize
from os import path

def read_wikitext(file_path):
    assert path.exists(file_path), '{} does not exist'.format(file_path)
    sents = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            if line.strip == '' or line.startswith(' = '):
                continue

            for sent in sent_tokenize(line):
                sents.append(sent.replace('<unk>', UNK_TAG))

    return sents