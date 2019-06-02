from flask_app.entrypoint import app

from flask_app.utils import get_json, jsonerror
from flask_app.utils.app_utils import get_config
from flask_app.nlu_main import nlu_load_pretrained
from flask import jsonify, request
from os import path
import sys, traceback

@app.route("/demo/cloze_predict", methods=['POST'])
def cloze_predict():
    try:
        content = get_json(request)
        model = nlu_load_pretrained('bert')
        
    except:
        traceback.print_exc(limit=2, file=sys.stdout)
        return jsonerror('Runtime exception encountered when handling request')
