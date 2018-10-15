from flask_app.nlu_main import nlu_init_model, nlu_predict, nlu_release_model
from flask_app.nlu_train import nlu_train_file
from config import UPLOAD_FOLDER, LOGS_FOLDER, CONFIG_PATH, BASE_PATH, PYTHON_PATH
from flask import request, flash, redirect, jsonify
from werkzeug.utils import secure_filename
from os import path, makedirs, remove
import time
import uuid
import json
import subprocess
import logging

consoleHandler = logging.StreamHandler()
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

TRAIN_PROCESSES = dict()

def allowed_file(filename, allowed_exts=['json']):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

def save_config(app):
    with open(CONFIG_PATH, 'w') as cfg_file:
        json.dump({
            'MODELS': app.config.get('MODELS', None),
        }, cfg_file)

def get_config(app):
    if path.isfile(CONFIG_PATH):
        try:
            cfg = json.load(open(CONFIG_PATH, 'r'))
            app.config.update(cfg)
        except:
            logging.warning('Failed to load configuration. Using defaults')
            pass

def delete_model(app, model_id):
    get_config(app)

    all_models = app.config['MODELS']

    if model_id not in all_models:
        return jsonerror('Model ID not found')
    else:
        model_config = all_models[model_id]

        if path.exists(model_config['CLF_MODEL_PATH']):
            remove(model_config['CLF_MODEL_PATH'])
        
        if path.exists(model_config['ENT_MODEL_PATH']):
            remove(model_config['ENT_MODEL_PATH'])

        del all_models[model_id]

        app.config['MODELS'] = all_models

        save_config(app)


def jsonerror(*args, **kwargs):
    response = jsonify(*args, **kwargs)
    response.status_code = 400
    return response

def initialize(app):
    app.config["SECRET_KEY"] = "test secret key".encode("utf8")
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['LOGS_FOLDER'] = LOGS_FOLDER

    get_config(app)

    # create directories if not exists
    if not path.exists(app.config['UPLOAD_FOLDER']):
        makedirs(app.config['UPLOAD_FOLDER'])
    
    if not path.exists(app.config['LOGS_FOLDER']):
        makedirs(app.config['LOGS_FOLDER'])

    @app.route("/")
    def index():
        return "Server is up and running! BASE_PATH: %s" % BASE_PATH

    @app.route("/upload", methods=['POST'])
    def upload():
        if request.method == 'POST':
            if 'file' not in request.files:
                return jsonerror('No file included')
            u_file = request.files['file']
            if u_file.filename == '':
                return jsonerror('No selected file')
            if u_file and allowed_file(u_file.filename):
                request_form = request.form

                callback_url = request_form.get('callback_url', '')
                if 'model_id' in request_form:
                    prev_model_id = request_form['model_id']

                    if prev_model_id in TRAIN_PROCESSES:
                        process = TRAIN_PROCESSES[prev_model_id]
                        return_code = process.poll()
                        if return_code is None: # process is still running
                            process.kill() # immediately kill the process

                    print('Deleting previous model', prev_model_id)
                    delete_model(app, prev_model_id)

                model_id = str(uuid.uuid4())

                filename = model_id + '_' + secure_filename(u_file.filename)
                save_path = path.join(app.config['UPLOAD_FOLDER'], filename)
                u_file.save(save_path)

                print('Upload complete. Beginning training model', model_id)

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
                    with open(log_file_name, 'w') as log_fp:
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

    @app.route("/status", methods=['POST'])
    def flask_get_status():
        content = request.get_json()

        if 'model_id' not in content:
            return jsonerror('Model ID must be provided')
        else:
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

    @app.route("/predict", methods=['POST'])
    def flask_predict():
        content = request.get_json()

        if 'query' not in content:
            return jsonerror('Invalid JSON object')
        elif 'model_id' not in content:
            return jsonerror('Model ID must be provided')
        else:
            get_config(app)
            
            model_count = len(list(app.config['MODELS'].keys())) \
                if 'MODELS' in app.config \
                else 0

            if model_count == 0:
                return jsonerror('No model trained')
            else:
                model_id = content['model_id'].strip()

                if model_id not in app.config['MODELS']:
                    return jsonerror('Model ID not found')
                else:
                    model_config = app.config['MODELS'][model_id]
                    nlu_init_model(
                        model_id,
                        model_config['CLF_MODEL_PATH'],
                        model_config['ENT_MODEL_PATH']
                    )
                    result = nlu_predict(model_id, content['query'])
                    return jsonify(result)
