from flask_app import app
from flask_app.utils import jsonerror, allowed_file
from flask_app.utils.app_utils import get_config, save_config
from flask import request, flash, redirect, jsonify
from werkzeug.utils import secure_filename
from flask_app.nlu_train import nlu_train_file
from flask_app.nlu_main import TRAIN_PROCESSES
from config import PYTHON_PATH
from os import path
import subprocess
import sys, traceback
import time
import uuid

@app.route("/upload", methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonerror('No file included')
        u_file = request.files['file']
        if u_file.filename == '':
            return jsonerror('No selected file')
        if not u_file or not allowed_file(u_file.filename):
            return jsonerror('Invalid file')

        request_form = request.form

        callback_url = request_form.get('callback_url', '')
        if 'model_id' in request_form:
            prev_model_id = request_form['model_id']

            if prev_model_id in TRAIN_PROCESSES:
                process = TRAIN_PROCESSES[prev_model_id]
                return_code = process.poll()
                if return_code is None: # process is still running
                    process.kill() # immediately kill the process

            # This deletes the model before training is complete so...

            app.logger.info('Deleting previous model', prev_model_id)
            # delete_model(app, prev_model_id)

        model_id = str(uuid.uuid4())

        filename = model_id + '_' + secure_filename(u_file.filename)
        save_path = path.join(app.config['UPLOAD_FOLDER'], filename)
        u_file.save(save_path)

        app.logger.info('Upload complete. Beginning training model', model_id)

        clf_model_path = save_path + '.cls.bin'
        ent_model_path = save_path + '.ent.bin'

        get_config(app)

        if not app.config.get('USE_QUEUE', True):
            clf_model_path, ent_model_path = nlu_train_file(model_id,
                save_path,
                clf_model_path,
                ent_model_path)
        else:
            log_file_name = path.join(app.config['LOGS_FOLDER'], model_id + '.log')
            with open(log_file_name, 'w', encoding='utf8') as log_fp:
                TRAIN_PROCESSES[model_id] = subprocess.Popen(
                    [
                        PYTHON_PATH, '-m', 'flask_app.nlu_train', 
                        '--model_id', model_id, 
                        '--save_path', save_path,
                        '--clf_model_path', clf_model_path,
                        '--ent_model_path', ent_model_path,
                        '--callback_url', callback_url
                    ],
                    stdout=log_fp
                )

        if app.config.get('MODELS', None) is None:
            app.config['MODELS'] = {}

        app.config['MODELS'][model_id] = {
            'CLF_MODEL_PATH': clf_model_path,
            'ENT_MODEL_PATH': ent_model_path,
            'created': time.time()
        }
        save_config(app)

        return jsonify({
            'model_id': model_id
        })
    except:
        traceback.print_exc(limit=2, file=sys.stdout)
        return jsonerror('Runtime exception encountered when handling request')
