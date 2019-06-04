from os import path

# from text_classification.fast_text.model import FastTextWrapper
# from text_classification.fast_text.train import FastTextLearner

from text_classification.ensemble.model import EnsembleWrapper
# from text_classification.ensemble.train import EnsembleLearner

from entities_recognition.transformer.model import TransformerSequenceTaggerWrapper
from common.utils import wordpunct_space_tokenize
# from entities_recognition.transformer.train import TransformerSequenceTaggerLearner

from flask_app.visualize import visualize_inputs

# from flask_app.entrypoint import app_cache

import logging
consoleHandler = logging.StreamHandler()
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

CLF_MODEL = {}
ENT_MODEL = {}

from sent_to_vec.masked_lm.bert_model import BertLMWrapper
from config import BASE_PATH
from common.utils import dotdict

PRETRAINED_MODELS = dotdict({
    'bert_vi_base': dotdict({
        "base_class": BertLMWrapper,
        "filename": 'bert_vi_base.bin',
        "loaded_object": None
    })
})

TRAIN_PROCESSES = dict()

# from pympler import muppy, summary
# all_objects = muppy.get_objects()

# sum1 = summary.summarize(all_objects)
# summary.print_(sum1)

def nlu_load_pretrained(model_name: str):
    global PRETRAINED_MODELS
    if model_name in PRETRAINED_MODELS:
        if PRETRAINED_MODELS[model_name].loaded_object is None:
            model = PRETRAINED_MODELS[model_name].base_class(
                from_fp=path.join(BASE_PATH, PRETRAINED_MODELS[model_name].filename)
            )
            model.init_model()

            logging.info('Pretrained model %s loaded' % model_name)
            PRETRAINED_MODELS[model_name].loaded_object = model

            return model
        else:
            return PRETRAINED_MODELS[model_name].loaded_object
    else:
        logging.error('Invalid model name')


def nlu_init_model(model_id, filename, ent_file_name):
    global CLF_MODEL, ENT_MODEL
    if model_id not in CLF_MODEL:
        logging.info('Loading models for id %s' % model_id)
        if filename is not None and filename != '' and path.exists(filename):
            CLF_MODEL[model_id] = EnsembleWrapper(from_fp=filename)
            CLF_MODEL[model_id].init_model()
            logging.info('Classification model loaded')
        else:
            logging.error('Classification model does not exist')
        
        if ent_file_name is not None and ent_file_name != '' and path.exists(ent_file_name):
            ENT_MODEL[model_id] = TransformerSequenceTaggerWrapper(from_fp=ent_file_name)
            ENT_MODEL[model_id].init_model()
            logging.info('Entity tagging model loaded')
        else:
            logging.error('Entity tagging model does not exist')

    # sum1 = summary.summarize(all_objects)
    # summary.print_(sum1)
    
def nlu_predict(model_id, query, contexts = None):
    if contexts is None:
        ret_intents = CLF_MODEL.get(model_id)([query])
    else:
        ret_intents = CLF_MODEL.get(model_id)([query], contexts=contexts)

    intents_result = {
        "intents": ret_intents[0] if ret_intents is not None else None
    }

    entities_result = {}
    if ENT_MODEL.get(model_id, None) is not None:
        ret_entities = ENT_MODEL[model_id]([wordpunct_space_tokenize(query)])
        entities_result = {
            "entities": ret_entities[0] if ret_entities is not None else None
        }

    result = {**intents_result, **entities_result}

    # sum1 = summary.summarize(all_objects)
    # summary.print_(sum1)

    return result

def nlu_visualize(query, model_id=None, n_clusters=None):
    if model_id is not None:
        return visualize_inputs(query, CLF_MODEL.get(model_id), n_clusters)
    else:
        return visualize_inputs(query, None, n_clusters)

def nlu_release_model(model_id):
    global CLF_MODEL, ENT_MODEL

    if model_id in CLF_MODEL:
        del CLF_MODEL[model_id] #.pop() also works but not as direct
    
    if model_id in ENT_MODEL:
        del ENT_MODEL[model_id]
