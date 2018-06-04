import json
from config import START_TAG, STOP_TAG
from common.utils import wordpunct_space_tokenize
from glove_utils import init_glove
from text_classification.fast_text.train import trainIters as clf_trainIters
from text_classification.fast_text.predict import predict as clf_predict, load_model as clf_load_model
from entities_recognition.bilstm.train import trainIters as ent_trainIters
from entities_recognition.bilstm.predict import predict as ent_predict, load_model as ent_load_model

IGNORE_CONTEXT = True  # flag for ignoring intents with contexts

CLF_MODEL = {}
CLF_CLASSES = {}
ENT_MODEL = {}
ENT_TAG_TO_IX = {}

init_glove()


def nlu_init_model(model_id, filename, ent_file_name):
    global CLF_MODEL, CLF_CLASSES, ENT_MODEL, ENT_TAG_TO_IX

    if model_id not in CLF_MODEL:
        CLF_MODEL[model_id], CLF_CLASSES[model_id] = clf_load_model(filename)
        if ent_file_name is not None and ent_file_name != '':
            ENT_MODEL[model_id], ENT_TAG_TO_IX[model_id] = ent_load_model(ent_file_name)


def nlu_predict(model_id, query):
    cls_probs, cls_idxs = clf_predict(CLF_MODEL[model_id], [query], k=5)[0]
    cls_probs = cls_probs.squeeze(0)
    cls_idxs = cls_idxs.squeeze(0)
    intents_result = {
        "intents": [
            {
                "intent": CLF_CLASSES[cls.item()],
                "confidence": cls_probs[idx].item()
            }
            for idx, cls in enumerate(cls_idxs)
        ]
    }

    entities_result = {}
    if ENT_MODEL.get(model_id, None) is not None and ENT_TAG_TO_IX.get(model_id, None) is not None:
        entities = ent_predict(ENT_MODEL[model_id], [query], ENT_TAG_TO_IX)
        entities_result = {"entities": entities[0]}

    result = {**intents_result, **entities_result}
    return result


def nlu_train_file(model_id, save_path, clf_model_path=None, ent_model_path=None):
    global CLF_MODEL, CLF_CLASSES, ENT_MODEL, ENT_TAG_TO_IX

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
            cls = classes.index(intent['name'])
            if len(examples) > 0:
                for example in examples:
                    if example['entities']:
                        text = ''.join([entity['text'] for entity in example['entities']])
                        example_tags = []
                        training_data.append((text, cls))

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
    CLF_MODEL[model_id] = clf_trainIters(training_data,
                                         classes,
                                         n_iters=50,
                                         log_every=10,
                                         verbose=1,
                                         learning_rate=1e-3,
                                         batch_size=64,
                                         save_path=clf_model_path)
    CLF_CLASSES[model_id] = classes

    if num_entities > 0:
        tag_names = list(set([START_TAG, STOP_TAG] + tag_names))
        tag_to_ix = {tag: idx for idx, tag in enumerate(tag_names)}

        ent_model_path = ent_model_path or save_path+'.ent.bin'

        print('Training entities recognition model')
        ENT_MODEL[model_id] = ent_trainIters(entities_data,
                                             tag_to_ix,
                                             n_iters=50,
                                             log_every=10,
                                             verbose=1,
                                             save_path=ent_model_path)
        ENT_TAG_TO_IX[model_id] = tag_to_ix

    return clf_model_path, ent_model_path

