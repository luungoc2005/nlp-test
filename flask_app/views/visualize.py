from flask_app.entrypoint import app

from flask_app.utils import get_json, jsonerror
from flask_app.utils.app_utils import get_config
from flask_app.nlu_main import TRAIN_PROCESSES, nlu_visualize, nlu_init_model
from flask import jsonify, request
from os import path
import sys, traceback

@app.route("/visualize", methods=['POST'])
def flask_visualize():
    try:
        content = get_json(request)

        if 'items' not in content:
            return jsonerror('Invalid JSON object')
        get_config(app)
        
        model_id = None
        if 'model_id' in content:
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
        result = nlu_visualize(content.get('items', []), model_id=model_id, n_clusters=n_clusters)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc(limit=2, file=sys.stderr)
        return jsonerror('Runtime exception encountered when handling request: %s' % str(e))