# from glove_utils import init_glove
from flask_app.nlu_main import nlu_init_model, nlu_predict, nlu_train_file
from config import UPLOAD_FOLDER, CONFIG_PATH
from flask import request, flash, redirect, jsonify
from werkzeug.utils import secure_filename
from os import path

import json


def allowed_file(filename, allowed_exts=['json']):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts


def save_config(app):
    with open(CONFIG_PATH, 'w') as cfg_file:
        json.dump({
            'MODEL_PATH': app.config.get('MODEL_PATH', None)
        }, cfg_file)


def initialize(app):
    # init_glove()
    app.config["SECRET_KEY"] = "test secret key".encode("utf8")
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    if path.isfile(CONFIG_PATH):
        cfg = json.load(open(CONFIG_PATH, 'r'))
        app.config.update(cfg)

        if app.config.get('MODEL_PATH', None) is not None:
            nlu_init_model(app.config['MODEL_PATH'])
            print('Model loaded successfully')

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
                filename = secure_filename(file.filename)
                save_path = path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                flash('Upload complete. Beginning training')
                nlu_train_file(save_path)

                save_config(app)
                return redirect(request.url)

    @app.route("/predict", methods=['POST'])
    def flask_predict():
        content = request.get_json()

        if 'query' not in content:
            flash('Invalid JSON object')
            return redirect(request.url)
        else:
            result = nlu_predict(content['query'])
            return jsonify(result)
