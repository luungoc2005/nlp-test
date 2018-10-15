from os import path

# from text_classification.fast_text.model import FastTextWrapper
# from text_classification.fast_text.train import FastTextLearner

from text_classification.ensemble.model import EnsembleWrapper
from text_classification.ensemble.train import EnsembleLearner

from entities_recognition.bilstm.model import SequenceTaggerWrapper
from entities_recognition.bilstm.train import SequenceTaggerLearner

import logging
consoleHandler = logging.StreamHandler()
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

CLF_MODEL = {}
ENT_MODEL = {}

# from pympler import muppy, summary
# all_objects = muppy.get_objects()

# sum1 = summary.summarize(all_objects)
# summary.print_(sum1)

def nlu_init_model(model_id, filename, ent_file_name):
    global CLF_MODEL, ENT_MODEL
    logging.info('Loading models for id %s' % model_id)
    if model_id not in CLF_MODEL:
        if filename is not None and filename != '' and path.exists(filename):
            CLF_MODEL[model_id] = EnsembleWrapper(from_fp=filename)
            CLF_MODEL[model_id].init_model()
            logging.info('Classification model loaded')
        else:
            logging.error('Classification model does not exist')
        
        if ent_file_name is not None and ent_file_name != '' and path.exists(ent_file_name):
            ENT_MODEL[model_id] = SequenceTaggerWrapper(from_fp=ent_file_name)
            ENT_MODEL[model_id].init_model()
            logging.info('Entity tagging model loaded')
        else:
            logging.error('Entity tagging model does not exist')

    # sum1 = summary.summarize(all_objects)
    # summary.print_(sum1)
    
def nlu_predict(model_id, query):
    intents_result = {
        "intents": CLF_MODEL.get(model_id)([query])[0]
    }

    entities_result = {}
    if ENT_MODEL.get(model_id, None) is not None and ENT_TAG_TO_IX.get(model_id, None) is not None:
        entities_result = {"entities": ENT_MODEL[model_id]([query])[0]}

    result = {**intents_result, **entities_result}

    # sum1 = summary.summarize(all_objects)
    # summary.print_(sum1)

    return result

def nlu_release_model(model_id):
    global CLF_MODEL, ENT_MODEL

    if model_id in CLF_MODEL:
        del CLF_MODEL[model_id] #.pop() also works but not as direct
    
    if model_id in ENT_MODEL:
        del ENT_MODEL[model_id]