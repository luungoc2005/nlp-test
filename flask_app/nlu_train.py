
import json
import torch
from os import path
from text_classification.ensemble.model import EnsembleWrapper
from text_classification.ensemble.train import EnsembleLearner
from entities_recognition.bilstm.model import SequenceTaggerWrapper
from entities_recognition.bilstm.train import SequenceTaggerLearner
from common.callbacks import EarlyStoppingCallback
import argparse
from datetime import datetime

IGNORE_CONTEXT = True  # flag for ignoring intents with contexts
CLF_MODEL = dict()
ENT_MODEL = dict()

def nlu_train_file(model_id, save_path, clf_model_path=None, ent_model_path=None):
    data = json.load(open(save_path, 'r'))
    print('Loaded %s intents' % len(data))

    classes = list(set([
        intent['name']
        for intent in data
        if (not IGNORE_CONTEXT or len(intent['inContexts']) == 0)
    ]))

    entities_data = []
    tag_names = []
    training_data = []

    for intent in data:
        examples = intent['examples']
        if intent['name'] in classes:
            if len(examples) > 0:
                for example in examples:
                    if example['entities']:
                        text = ''.join([entity['text'] for entity in example['entities']])
                        example_tags = []
                        training_data.append((text, intent['name']))

                        entities = [
                            entity for entity in example['entities']
                            if entity.get('nlpEntityId', 0) != 0
                            and entity.get('name', '') != ''
                        ]

                        if len(entities) > 0:
                            for entity in example['entities']:
                                if entity.get('nlpEntityId', 0) != 0 and \
                                   entity.get('name', '') != '':
                                    example_tags.extend([
                                        'B-' + entity.get('name') if idx == 0 else 'I-' + entity.get('name')
                                        for idx, _ in enumerate(wordpunct_space_tokenize(entity.get('text', '')))
                                    ])
                                    tag_names.append(entity.get('name'))
                                else:
                                    example_tags.extend(['-' for _ in wordpunct_space_tokenize(entity.get('text'))])
                            entities_data.append(text, example_tags.join(' '))

    num_entities = len(set(tag_names))
    print('Loaded %s examples; %s unique entities' % (len(training_data), num_entities))

    clf_model_path = clf_model_path or save_path+'.clf.bin'
    ent_model_path = ent_model_path or ''

    print('Training classification model')

    CLF_MODEL[model_id] = EnsembleWrapper()
    clf_learner = EnsembleLearner(CLF_MODEL[model_id])
    clf_learner.fit(
        training_data=training_data,
        batch_size=64
    )
    torch.save(CLF_MODEL[model_id].get_state_dict(), clf_model_path)

    if num_entities > 0:
        tag_names = list(set([START_TAG, STOP_TAG] + tag_names))
        tag_to_ix = {tag: idx for idx, tag in enumerate(tag_names)}

        ent_model_path = ent_model_path or save_path+'.ent.bin'

        print('Training entities recognition model')
        ENT_MODEL[model_id] = SequenceTaggerWrapper({'tag_to_ix': tag_to_ix})
        ent_learner = SequenceTaggerLearner(ENT_MODEL[model_id])
        learner.fit(
            training_data=entities_data,
            epochs=300,
            callbacks=[EarlyStoppingCallback()]
        )
        torch.save(ENT_MODEL[model_id].get_state_dict(), ent_model_path)

    return clf_model_path, ent_model_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default='')
    parser.add_argument("--save_path", type=str, default='')
    parser.add_argument("--clf_model_path", type=str, default='')
    parser.add_argument("--ent_model_path", type=str, default='')

    args = parser.parse_args()

    print(args)

    if args.model_id == '':
        print('Training failed: model_id is null')
        exit()

    if not path.exists(args.save_path):
        print('Training failed: save_path does not exist')
        exit()

    if args.clf_model_path == '' or args.ent_model_path == '':
        print('Training failed: model path is null')
        exit()

    print('Training started at %s' % str(datetime.now()))
    nlu_train_file(args.model_id, args.save_path, args.clf_model_path, args.ent_model_path)
    print('Training finished at %s' % str(datetime.now()))