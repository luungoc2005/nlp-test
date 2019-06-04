from flask_app.entrypoint import app

from flask_app.utils import get_json, jsonerror
from flask_app.utils.app_utils import get_config
from flask import jsonify, request
from os import path
from common.utils import word_tokenize
import sys, traceback
import logging

LANGUAGE_MODEL = None

@app.route("/demo/tokenize", methods=['POST'])
def demo_tokenize():
    try:
        content = get_json(request)

        if 'items' not in content:
            return jsonerror('Invalid JSON object')
        
        items = content.get('items', [])

        result = [word_tokenize(sent) for sent in items]
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(traceback.print_exc(limit=5))
        return jsonerror('Runtime exception encountered when handling request: %s' % str(e))
