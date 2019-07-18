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

@app.route("/demo/sentiment_predict", methods=['POST'])
def demo_sentiment_predict():
    try:
        content = get_json(request)

        if 'items' not in content:
            return jsonerror('Invalid JSON object')
        
        items = content.get('items', [])

        language = content.get('language', 'en')
        if language == 'en':
            model = nlu_load_pretrained('bert_en_sentiment')
        elif language == 'vi':
            model = nlu_load_pretrained('bert_vi_sentiment')
        else:
            raise ValueError('Unsupported language code')

        print(model)
        predicted_items = model(items)
        
        score = 0
        print(predicted_items[0])
        for intent in predicted_items[0]:
            if intent['intent'] == 'positive':
                score += 1 * intent['confidence']
            elif intent['intent'] == 'negative':
                score -= 1 * intent['confidence']

        return jsonify({ 'scores': [score] })

    except Exception as e:
        logging.error(traceback.print_exc(limit=5))
        return jsonerror('Runtime exception encountered when handling request: %s' % str(e))
