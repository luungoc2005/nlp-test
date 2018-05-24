import json
from glove_utils import init_glove
from text_classification.fast_text.train import trainIters
from text_classification.fast_text.predict import predict, load_model

IGNORE_CONTEXT = True  # flag for ignoring intents with contexts
MODEL = None
CLASSES = None


init_glove()


def nlu_init_model(filename):
    global MODEL, CLASSES
    MODEL, CLASSES = load_model(filename)


def nlu_predict(query):
    probs, idxs = predict(MODEL, [query], k=5)[0]
    probs = probs.squeeze(0)
    idxs = idxs.squeeze(0)
    result = {
        "intents": [
            {
                "intent": CLASSES[cls.item()],
                "confidence": probs[idx].item()
            }
            for idx, cls in enumerate(idxs)
        ]
    }
    return result


def nlu_train_file(save_path):
    data = json.load(open(save_path, 'r'))
    print('Loaded %s intents' % len(data))

    classes = list(set([
        intent['name']
        for intent in data
        if (not IGNORE_CONTEXT or len(intent['inContexts']) == 0)
    ]))

    training_data = []

    for intent in data:
        examples = intent['examples']
        if intent['name'] in classes:
            cls = classes.index(intent['name'])
            if len(examples) > 0:
                for example in examples:
                    text = ''.join([entity['text'] for entity in example['entities']])
                    training_data.append((text, cls))

    print('Loaded %s examples' % len(training_data))
    model_path = save_path+'.bin'

    model = trainIters(training_data,
                       classes,
                       n_iters=50,
                       log_every=10,
                       verbose=1,
                       learning_rate=1e-3,
                       batch_size=64,
                       save_path=save_path+'.bin')
    return model, classes, model_path
