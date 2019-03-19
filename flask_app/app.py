from flask import Flask
from flask_app.initialize import initialize
from os import environ
import argparse
import time
import logging

app = Flask(__name__)

def start_server(args=None):
    try:
        gunicorn_logger = logging.getLogger('gunicorn.error')
        app.logger.handlers = gunicorn_logger.handlers
        app.logger.setLevel(gunicorn_logger.level)
    except:
        pass

    initialize(app)
    from flask_app.views import index, upload, delete, status, predict, visualize

    if args is not None:
        if args.cors:
            try:
                from flask_cors import CORS
                CORS(app)
            except:
                pass

        if args.debug:
            app.run(processes=1, debug=True, threaded=False)
        else:
            app.config['USE_QUEUE'] = args.queue
            app.run()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--queue", type=bool, default=False)
    parser.add_argument("--cors", action='store_true', default=False)
    args = parser.parse_args()

    start_server(args)
