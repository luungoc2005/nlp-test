from config import UPLOAD_FOLDER, LOGS_FOLDER, CONFIG_PATH, BASE_PATH
from flask import request, flash, redirect, jsonify
from flask_app.utils.app_utils import get_config
from os import path, makedirs, remove
import time
import uuid
import json

# consoleHandler = logging.StreamHandler()
# logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler())

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