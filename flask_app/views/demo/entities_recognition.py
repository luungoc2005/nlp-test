from flask_app.entrypoint import app

from flask_app.utils import get_json, jsonerror
from flask_app.utils.app_utils import get_config
from flask_app.nlu_main import nlu_load_pretrained
from flask import jsonify, request
from config import MASK_TAG, START_TAG, STOP_TAG
from common.utils import wordpunct_space_tokenize

import torch

from os import path
import logging
import sys, traceback

@app.route("/demo/entities_predict", methods=['POST'])
def demo_entities_predict():
    try:
        content = get_json(request)

        if 'items' not in content:
            return jsonerror('Invalid JSON object')
        
        items = content.get('items', [])
        items = [wordpunct_space_tokenize(sent) for sent in items]

        model = nlu_load_pretrained('lstm_en_tagger')

        tag_seq_batch, _, sent_batch = model(items, return_logits=True)

        return jsonify([[
            {
                'tag': model.ix_to_tag.get(tag_seq_batch[sent_ix][word_ix]),
                'label': word
            }
            for word_ix, word in enumerate(sent)
        ] 
        for sent_ix, sent in enumerate(sent_batch)])
        
    except Exception as e:
        logging.error(traceback.print_exc(limit=5))
        return jsonerror('Runtime exception encountered when handling request: %s' % str(e))
