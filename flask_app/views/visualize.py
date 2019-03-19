from flask_app.entrypoint import app

from flask_app.utils import get_json, jsonerror
from flask_app.utils.app_utils import get_config
from flask_app.nlu_main import TRAIN_PROCESSES, nlu_visualize
from flask import jsonify, request
from os import path
import sys, traceback

@app.route("/visualize", methods=['POST'])
def flask_visualize():
    try:
        content = get_json(request)

        if 'items' not in content:
            return jsonerror('Invalid JSON object')
        elif 'model_id' not in content:
            return jsonerror('Model ID must be provided')
        
        get_config(app)
        
        model_count = len(list(app.config['MODELS'].keys())) \
            if 'MODELS' in app.config \
            else 0

        if model_count == 0:
            return jsonerror('No model trained')
        
        model_id = content['model_id'].strip()

        if model_id not in app.config['MODELS']:
            return jsonerror('Model ID not found')
        
        model_config = app.config['MODELS'][model_id]
        nlu_init_model(
            model_id,
            model_config['CLF_MODEL_PATH'],
            model_config['ENT_MODEL_PATH']
        )
        n_clusters = content.get('n_clusters', None)
        result = nlu_visualize(model_id, content.get('items', []), n_clusters)
        return jsonify(result)
    except:
        traceback.print_exc(limit=2, file=sys.stdout)
        return jsonerror('Runtime exception encountered when handling request')
