from flask import Flask
from flask_app.initialize import initialize
from os import environ
import argparse
import time
import logging

app = Flask(__name__)
initialize(app)

try:
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
except:
    pass

# from werkzeug.contrib.cache import SimpleCache
# app_cache = SimpleCache()

from flask_app.views import \
    index, \
    upload, \
    delete, \
    status, \
    predict, \
    visualize

from flask_app.views.demo import \
    tokenize, \
    cloze_predict, \
    language_identification, \
    entities_recognition, \
    pos_predict

from flask_cors import CORS
CORS(app)

def apply_args(cli_args=None):
    if cli_args is not None:
        if cli_args.cors:
            try:
                from flask_cors import CORS
                CORS(app)
            except:
                pass

    app.config['USE_QUEUE'] = cli_args.queue

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--queue", type=bool, default=False)
    parser.add_argument("--cors", action='store_true', default=False)
    args = parser.parse_args()
    
    apply_args(args)

    if args.debug or environ.get('DEBUG', '') != '':
        app.config['DEBUG'] = True
        app.run(processes=1, debug=True, threaded=False, host='0.0.0.0', port=5000)
    else:
        app.run()
