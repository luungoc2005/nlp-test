import torch
import argparse
import io
import json
from entities_recognition.bilstm.model import *

# Usage:
# python conll_eval.py entities_recognition/bilstm-rnn-conll2003-vanilla.bin entities_recognition/tag_to_ix.json data/CoNLL-2003/eng.testa > testa.out.txt
# conlleval < testa.out.txt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model', help='Path to model'
    )
    parser.add_argument(
        'tag_to_ix', help='Path to tag_to_ix file', default=''
    )
    parser.add_argument(
        'input', help='Path to CoNLL test file', default=''
    )

    args = parser.parse_args()

    tag_to_ix = json.load(open(args.tag_to_ix, 'r'))

    model = BiLSTM_CRF(tag_to_ix)
    model.load_state_dict(torch.load(args.model))
    
    ix_to_tag = {value: key for key, value in model.tag_to_ix.items()}

    fin = io.open(args.input, 'r', encoding='utf-8', newline='\n', errors='ignore')

    current_txt = []
    current_pos = []
    current_tags = []
    tagset = []
    for line in fin:
        line = line.strip()
        if len(line) > 0: # skip blank lines
            tmp = line.split(' ')
            if tmp[0] != '-DOCSTART-':
                current_txt.append(tmp[0])
                current_pos.append(tmp[1])
                current_tags.append(tmp[-1])
            else:
                print(line)
        else:
            if len(current_txt) > 0:
                _, tag_seq = model(current_txt)
                tag_seq = [ix_to_tag[tag] for tag in tag_seq]
                for idx, token in enumerate(current_txt):
                    print('%s %s %s %s' % (token, current_pos[idx], current_tags[idx], tag_seq[idx]))
                current_txt = []
                current_tags = []
                current_pos = []
            print('')
    fin.close()