# from glove_utils import init_glove
from config import UPLOAD_FOLDER, CONFIG_PATH
from text_classification.fast_text.train import trainIters
from text_classification.fast_text.predict import predict, load_model

from flask import request, flash, redirect, jsonify
from werkzeug.utils import secure_filename
from os import path

import json


def allowed_file(filename, allowed_exts = ['json']):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts


def train_file(save_path):
    data = json.load(open(save_path, 'r'))
    print('Loaded %s intents' % len(data))

    IGNORE_CONTEXT = True  # flag for ignoring intents with contexts

    classes = list(set([
        intent['name']
        for intent in data
        if (not IGNORE_CONTEXT or len(intent['inContexts']) == 0)
    ]))

    training_data = []

    for intent in data:
        examples = intent['examples']
        if intent['name'] in classes:
            cls = classes.index(intent['name'])
            if len(examples) > 0:
                for example in examples:
                    text = ''.join([entity['text'] for entity in example['entities']])
                    training_data.append((text, cls))

    print('Loaded %s examples' % len(training_data))
    model_path = save_path+'.bin'

    model = trainIters(training_data,
                       classes,
                       n_iters=50,
                       log_every=10,
                       verbose=1,
                       learning_rate=1e-3,
                       batch_size=64,
                       save_path=save_path+'.bin')
    return model, classes, model_path


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
            model, classes = load_model(app.config['MODEL_PATH'])
            app.config['MODEL'] = model
            app.config['CLASSES'] = classes
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

                model, classes, model_path = train_file(save_path)
                app.config['MODEL'] = model
                app.config['CLASSES'] = classes
                app.config['MODEL_PATH'] = model_path

                save_config(app)
                return redirect(request.url)

    @app.route("/predict", methods=['POST'])
    def flask_predict():
        if 'MODEL' not in app.config or 'CLASSES' not in app.config:
            flash('No model loaded')
            return redirect(request.url)
        else:
            content = request.get_json()
            model = app.config['MODEL']
            classes = app.config['CLASSES']

            if 'query' not in content:
                flash('Invalid JSON object')
                return redirect(request.url)
            else:
                result = predict(model, [content['query']])
                result = [(classes[val], idx) for (val, idx) in result]
                return jsonify(result)
