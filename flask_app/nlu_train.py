
import json
import torch
from os import path

from text_classification.fast_text.model import FastTextWrapper
from text_classification.fast_text.train import FastTextLearner

from text_classification.ensemble.model import EnsembleWrapper
from text_classification.ensemble.train import EnsembleLearner

from entities_recognition.bilstm.model import SequenceTaggerWrapper
from entities_recognition.bilstm.train import SequenceTaggerLearner
from common.callbacks import EarlyStoppingCallback, PrintLoggerCallback
import argparse
from datetime import datetime

import logging
consoleHandler = logging.StreamHandler()
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

IGNORE_CONTEXT = True  # flag for ignoring intents with contexts
CLF_MODEL = dict()
ENT_MODEL = dict()

# from pympler import muppy, summary
# all_objects = muppy.get_objects()

# sum1 = summary.summarize(all_objects)
# summary.print_(sum1)

def nlu_train_file(model_id, save_path, clf_model_path=None, ent_model_path=None):
    data = json.load(open(save_path, 'r'))
    logging.info('Loaded %s intents' % len(data))

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
    logging.info('Loaded %s examples; %s unique entities' % (len(training_data), num_entities))

    clf_model_path = clf_model_path or save_path+'.clf.bin'
    ent_model_path = ent_model_path or ''

    logging.info('Training classification model')

    CLF_MODEL[model_id] = EnsembleWrapper()
    clf_learner = EnsembleLearner(CLF_MODEL[model_id])
    clf_learner.fit(
        training_data=training_data,
        batch_size=64
        # epochs=300
        # callbacks=[PrintLoggerCallback(), EarlyStoppingCallback()]
    )
    CLF_MODEL[model_id].save(clf_model_path)

    if num_entities > 0:
        tag_names = list(set([START_TAG, STOP_TAG] + tag_names))
        tag_to_ix = {tag: idx for idx, tag in enumerate(tag_names)}

        ent_model_path = ent_model_path or save_path+'.ent.bin'

        logging.info('Training entities recognition model')
        ENT_MODEL[model_id] = SequenceTaggerWrapper({'tag_to_ix': tag_to_ix})
        ent_learner = SequenceTaggerLearner(ENT_MODEL[model_id])
        learner.fit(
            training_data=entities_data,
            epochs=300,
            callbacks=[PrintLoggerCallback(), EarlyStoppingCallback()]
        )
        ENT_MODEL[model_id].save(ent_model_path)

    return clf_model_path, ent_model_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default='')
    parser.add_argument("--save_path", type=str, default='')
    parser.add_argument("--clf_model_path", type=str, default='')
    parser.add_argument("--ent_model_path", type=str, default='')

    args = parser.parse_args()

    logging.info(args)

    if args.model_id == '':
        logging.error('Training failed: model_id is null')
        exit()

    if not path.exists(args.save_path):
        logging.error('Training failed: save_path does not exist')
        exit()

    if args.clf_model_path == '' or args.ent_model_path == '':
        logging.error('Training failed: model path is null')
        exit()

    logging.info('Training started at %s' % str(datetime.now()))
    nlu_train_file(args.model_id, args.save_path, args.clf_model_path, args.ent_model_path)
    logging.info('Training finished at %s' % str(datetime.now()))

    # sum1 = summary.summarize(all_objects)
    # summary.print_(sum1)