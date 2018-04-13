import torch
import string

from os import path, getcwd

from config import START_TAG, STOP_TAG
from entities_recognition.bilstm.model import BiLSTM_CRF
from entities_recognition.bilstm.train import SAVE_PATH
from common.utils import wordpunct_space_tokenize

def load_model(tag_to_ix):
    model = BiLSTM_CRF(tag_to_ix)
    model.load_state_dict(torch.load(SAVE_PATH))
    return model

def predict(model, input_data, tag_to_ix):
    # Invert the tag dictionary
    ix_to_tag = {value: key for key, value in tag_to_ix.items()}

    result = []
    print ('Raw predicted tags:')
    for sentence in input_data:
        tokens_in = wordpunct_space_tokenize(sentence)
        _, tag_seq = model(tokens_in)

        entities = {}
        entity_name = ''
        buffer = []
        print(tag_seq)

        for idx, tag in enumerate(tag_seq):
            tag_name = ix_to_tag[tag]

            if len(tag_name) > 2 and tag_name[:2] in ['B-', 'I-', 'O-']:
                new_entity_name = tag_name[2:]
                if entity_name != '' and \
                    (tag_name[:2] == 'B-' or \
                    entity_name != new_entity_name):
                    # Flush the previous entity
                    if not entity_name in entities:
                        entities[entity_name] = []
                        entities[entity_name].append(''.join(buffer))
                        buffer = []

                entity_name = new_entity_name
            
            # If idx is currently inside a tag
            if entity_name != '':
                buffer.append(tokens_in[idx])

                # Going outside the tag
                if idx == len(tag_seq) - 1 or \
                    tag_name == '-' or \
                    tag_name[:2] == 'O-':

                    if not entity_name in entities:
                        entities[entity_name] = []
                    entities[entity_name].append(''.join(buffer))
                    buffer = []
                    entity_name = ''

        result.append(entities)

    print('\n---')
    # Print results:
    for idx, sentence in enumerate(input_data):
        print('Input: %s' % sentence)
        print('Output: \n%s' % str(result[idx]))
        print ('')
    
    return result