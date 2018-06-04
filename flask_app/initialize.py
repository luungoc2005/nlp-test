# from glove_utils import init_glove
from flask_app.nlu_main import nlu_init_model, nlu_predict, nlu_train_file
from config import UPLOAD_FOLDER, CONFIG_PATH
from flask import request, flash, redirect, jsonify
from werkzeug.utils import secure_filename
from os import path
import uuid
import json


def allowed_file(filename, allowed_exts=['json']):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts


def save_config(app):
    with open(CONFIG_PATH, 'w') as cfg_file:
        json.dump({
            'MODELS': app.config.get('MODELS', None),
        }, cfg_file)


def initialize(app):
    # init_glove()
    app.config["SECRET_KEY"] = "test secret key".encode("utf8")
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    if path.isfile(CONFIG_PATH):
        cfg = json.load(open(CONFIG_PATH, 'r'))
        app.config.update(cfg)

    @app.route("/")
    def index():
        return "Server is up and running!"

    @app.route("/upload", methods=['POST'])
    def upload():
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file included')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                model_id = uuid.uuid4()

                filename = model_id + '_' + secure_filename(file.filename)
                save_path = path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)

                flash('Upload complete. Beginning training')

                clf_model_path = save_path
                ent_model_path = save_path

                if not app.config['USE_QUEUE']:
                    clf_model_path, ent_model_path = nlu_train_file(model_id,
                                                                    save_path,
                                                                    clf_model_path,
                                                                    ent_model_path)
                else:
                    pass

                if app.config.get('MODELS', None) is None:
                    app.config['MODELS'] = {}

                app.config['MODELS'][model_id] = {
                    'CLF_MODEL_PATH': clf_model_path,
                    'ENT_MODEL_PATH': ent_model_path
                }
                save_config(app)

                return redirect(request.url)

    @app.route("/predict", methods=['POST'])
    def flask_predict():
        content = request.get_json()

        if 'query' not in content:
            flash('Invalid JSON object')
            return redirect(request.url)
        else:
            model_count = len(list(app.config['MODELS'].keys())) \
                if 'MODELS' in app.config \
                else 0

            if model_count == 0:
                flash('No model trained')
                return redirect(request.url)
            else:
                # Use the first ID for default model id. For easy testing only
                model_id = content.get('model_id', next(app.config['MODELS'].keys()))

                if model_id not in app.config['MODELS']:
                    flash('Model id not found')
                else:
                    nlu_init_model(model_id,
                                   app.config['MODELS'][model_id]['CLF_MODEL_PATH'],
                                   app.config['MODELS'][model_id]['ENT_MODEL_PATH'])
                    result = nlu_predict(model_id, content['query'])
                    return jsonify(result)
