from flask_app.entrypoint import app

from flask_app.utils import get_json, jsonerror
from flask import jsonify, request
from config import BASE_PATH

from os import path
import sys, traceback
import logging

FASTTEXT_MODEL = None
def load_lid_model():
    global FASTTEXT_MODEL
    
    if FASTTEXT_MODEL is None:
        import fasttext
        FASTTEXT_MODEL = fasttext.load_model(
            path.join(BASE_PATH, 'lid.176.bin')
        )
        logging.info('FastText model loaded')
    
    return FASTTEXT_MODEL

@app.route("/demo/language_predict", methods=['POST'])
def demo_language_predict():
    try:
        content = get_json(request)
        
        if 'items' not in content:
            return jsonerror('Invalid JSON object')
        
        items = content.get('items', [])

        if isinstance(items, str):
            items = [items]

        pred, probs = load_lid_model().predict(items)
        
        result = []
        for ix in range(len(items)):
            result.append({
                'language': pred[ix][0][-2:],
                'probability': probs[ix][0]
            })
        return jsonify(result)

    except Exception as e:
        logging.error(traceback.print_exc(limit=5))
        return jsonerror('Runtime exception encountered when handling request: %s' % str(e))
