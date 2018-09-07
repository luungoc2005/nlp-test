from os import path
from text_classification.ensemble.model import EnsembleWrapper
from text_classification.ensemble.train import EnsembleLearner
from entities_recognition.bilstm.model import SequenceTaggerWrapper
from entities_recognition.bilstm.train import SequenceTaggerLearner

CLF_MODEL = {}
ENT_MODEL = {}

def nlu_init_model(model_id, filename, ent_file_name):
    global CLF_MODEL, ENT_MODEL
    print('Loading models for id %s' % model_id)
    if model_id not in CLF_MODEL:
        if filename is not None and filename != '' and path.exists(filename):
            CLF_MODEL[model_id] = EnsembleWrapper(from_fp=filename)
            CLF_MODEL[model_id].init_model()
            print('Classification model loaded')
        else:
            print('Classification model does not exist')
        
        if ent_file_name is not None and ent_file_name != '' and path.exists(ent_file_name):
            ENT_MODEL[model_id] = SequenceTaggerWrapper(from_fp=ent_file_name)
            ENT_MODEL[model_id].init_model()
            print('Entity tagging model loaded')
        else:
            print('Entity tagging model does not exist')

def nlu_predict(model_id, query):
    intents_result = {
        "intents": CLF_MODEL.get(model_id)([query])[0]
    }

    entities_result = {}
    if ENT_MODEL.get(model_id, None) is not None and ENT_TAG_TO_IX.get(model_id, None) is not None:
        entities_result = {"entities": ENT_MODEL[model_id]([query])[0]}

    result = {**intents_result, **entities_result}
    return result

