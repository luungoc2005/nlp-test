from flask_app.entrypoint import app

from flask_app.utils import get_json, jsonerror
from flask_app.utils.app_utils import get_config
from flask_app.nlu_main import nlu_load_pretrained
from flask import jsonify, request
from config import MASK_TAG, START_TAG, STOP_TAG

import torch

from os import path
import logging
import sys, traceback

@app.route("/demo/cloze_predict", methods=['POST'])
def demo_cloze_predict():
    try:
        content = get_json(request)
        model = nlu_load_pretrained('bert_vi_base')

        content = get_json(request)
        raw_line = [
            [START_TAG] + [
                tokenObject['value'] if not tokenObject['isMasked'] else MASK_TAG
                for tokenObject in content
            ] + [STOP_TAG]
        ]
        print(raw_line)
        # inputs = model.featurizer.transform(line)
        result = model(raw_line)[0]
        print(result.size())

        ix_to_word = model.featurizer.tokenizer.ix_to_word

        result = torch.topk(result, k=5, dim=1)

        values = result[0]
        indices = result[1]
        values = [
            [float(tensor) for tensor in word_probs]
            for word_probs in values
        ]
        labels = [
            [ix_to_word[int(idx)] if idx != 0 else '' for idx in word_result] 
            for word_result in indices
        ]
        # remove first & last items (start & stop tokens)
        values = values[1:-1]
        labels = labels[1:-1]

        formatted_result = {
            'values': values,
            'labels': labels
        }

        return jsonify(formatted_result)
        
    except Exception as e:
        logging.error(traceback.print_exc(limit=5))
        return jsonerror('Runtime exception encountered when handling request: %s' % str(e))
