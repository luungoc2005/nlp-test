from flask_app.app import app
from flask_app.utils import get_json, jsonerror
from flask_app.utils.app_utils import get_config
from flask_app.nlu_main import TRAIN_PROCESSES
from flask import jsonify
from os import path
import sys, traceback

@app.route("/status", methods=['POST'])
def flask_get_status():
    try:
        content = get_json(request)

        if 'model_id' not in content:
            return jsonerror('Model ID must be provided')
        
        get_config(app)

        model_id = content['model_id']

        if model_id not in app.config['MODELS']:
            return jsonerror('Model ID not found')

        model_config = app.config['MODELS'][model_id]

        if model_id in TRAIN_PROCESSES:
            process = TRAIN_PROCESSES[model_id]
            return_code = process.poll()
            if return_code is None:
                return jsonify({
                    'status': 'training'
                })
            elif return_code < 0:
                return jsonify({
                    'status': 'failed',
                    'error_code': return_code
                })
                
        if path.exists(model_config['CLF_MODEL_PATH']):
            return jsonify({
                'status': 'completed'
            })
        else:
            return jsonify({
                'status': 'not_found'
            })
    except:
        traceback.print_exc(limit=2, file=sys.stdout)
        return jsonerror('Runtime exception encountered when handling request')
