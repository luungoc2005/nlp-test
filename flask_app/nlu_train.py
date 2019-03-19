
import json
from os import path

# from text_classification.fast_text.model import FastTextWrapper
# from text_classification.fast_text.train import FastTextLearner

from text_classification.ensemble.model import EnsembleWrapper
from text_classification.ensemble.train import EnsembleLearner

from entities_recognition.transformer.model import TransformerSequenceTaggerWrapper
from entities_recognition.transformer.train import TransformerSequenceTaggerLearner
from entities_recognition.transformer.data import TransformerEntitiesRecognitionDataset

from common.callbacks import EarlyStoppingCallback, PrintLoggerCallback, ReduceLROnPlateau

from common.utils import wordpunct_space_tokenize
from config import START_TAG, STOP_TAG, EMPTY_TAG

import argparse
from datetime import datetime

IGNORE_CONTEXT = False  # flag for ignoring intents with contexts
CLF_MODEL = dict()
ENT_MODEL = dict()

# from pympler import muppy, summary
# all_objects = muppy.get_objects()

# sum1 = summary.summarize(all_objects)
# summary.print_(sum1)

def nlu_train_file(model_id, save_path, clf_model_path=None, ent_model_path=None):
    data = json.load(open(save_path, 'r', encoding='utf8'))
    print('Loaded %s intents' % len(data))

    classes = list(set([
        (intent['name'])
        for intent in data
        # if (not IGNORE_CONTEXT or len(intent['inContexts']) == 0)
    ]))
    contexts = {}
    if not IGNORE_CONTEXT:
        for class_name in classes:
            class_contexts = []
            class_intents = [
                intent for intent in data 
                if intent['name'] == class_name
            ]
            for intent in class_intents:
                if 'inContexts' in intent and isinstance(intent['inContexts'], list):
                    class_contexts.extend([
                        context['name'] 
                        for context in intent['inContexts']
                        if 'name' in context
                    ])
            contexts[intent['name']] = class_contexts

    entities_data = []
    tag_names = []
    training_data = []

    for intent in data:
        examples = intent['examples']
        if intent['name'] in classes:
            if len(examples) > 0:
                for example in examples:
                    if example['entities']:
                        text = ' '.join([entity['text'].strip() for entity in example['entities']])
                        example_tags = []
                        training_data.append((text, intent['name']))

                        entities = [
                            entity for entity in example['entities']
                            if entity.get('nlpEntityId', 0) != 0
                            and entity.get('name', '') != ''
                        ]

                        if len(entities) > 0:
                            for e_idx, entity in enumerate(example['entities']):
                                if entity.get('nlpEntityId', 0) != 0 and entity.get('name', '') != '':
                                    b_tag = 'B-' + entity.get('name')
                                    i_tag = 'I-' + entity.get('name')
                                    example_tags.extend([
                                        b_tag if idx == 0 else i_tag
                                        for idx, _ in enumerate(wordpunct_space_tokenize(entity.get('text', '')))
                                    ])
                                    tag_names.append(b_tag)
                                    tag_names.append(i_tag)
                                else:
                                    example_tags.extend([EMPTY_TAG for _ in wordpunct_space_tokenize(entity.get('text'))])
                                # add a tag for space
                                if (e_idx < len(example['entities']) - 1):
                                    example_tags.append(EMPTY_TAG)

                            entities_data.append((text, ' '.join(example_tags)))

    num_entities = len(set(tag_names))
    print('Loaded %s examples; %s unique entities' % (len(training_data), num_entities))

    clf_model_path = clf_model_path or save_path+'.clf.bin'

    print('Training classification model')

    CLF_MODEL[model_id] = EnsembleWrapper({'contexts': contexts})
    clf_learner = EnsembleLearner(CLF_MODEL[model_id])
    clf_learner.fit(
        training_data=training_data,
        batch_size=64
        # epochs=300
        # callbacks=[PrintLoggerCallback(), EarlyStoppingCallback()]
    )
    CLF_MODEL[model_id].save(clf_model_path)

    if num_entities > 0:
        tag_names = [EMPTY_TAG] + list(set(tag_names))
        tag_to_ix = {tag: idx + 1 for idx, tag in enumerate(tag_names)}
        print(tag_to_ix)
        print(entities_data)

        ent_model_path = ent_model_path or save_path+'.ent.bin'

        # print(entities_data)
        print('Training entities recognition model')
        ENT_MODEL[model_id] = TransformerSequenceTaggerWrapper(
        {
            'tag_to_ix': tag_to_ix,
            'char_embedding_dim': 50,
            'hidden_size': 350,
            'num_hidden_layers': 2,
            'num_attention_heads': 10,
            'intermediate_size': 512,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 256,
            'featurizer_seq_len': 256, # same as above
            'initializer_range': 0.02,
        })
        ent_learner = TransformerSequenceTaggerLearner(ENT_MODEL[model_id])
        entities_dataset = TransformerEntitiesRecognitionDataset(entities_data, tag_to_ix)
        ent_learner.fit(
            training_data=entities_dataset,
            batch_size=min(2, len(entities_data)),
            epochs=300,
            callbacks=[
                PrintLoggerCallback(),
                ReduceLROnPlateau(reduce_factor=4, patience=2),
                EarlyStoppingCallback(patience=5)
            ]
        )
        ENT_MODEL[model_id].save(ent_model_path)

    return clf_model_path, ent_model_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default='')
    parser.add_argument("--save_path", type=str, default='')
    parser.add_argument("--clf_model_path", type=str, default='')
    parser.add_argument("--ent_model_path", type=str, default='')
    parser.add_argument("--callback_url", type=str, default='')

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

    if args.callback_url.strip() != '':
        from urllib import request, parse
        qs = parse.urlencode({'model_id': args.model_id})
        full_url = '{}?{}'.format(args.callback_url, qs)
        print('Sending POST request to', full_url)
        request_obj = request.Request(
            full_url, 
            data=b'',
            headers={
                'Accept': 'text/html',
                'User-Agent': 'Mozilla/5.0'
            }
        )
        request.urlopen(request_obj)

    # sum1 = summary.summarize(all_objects)
    # summary.print_(sum1)