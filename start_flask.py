from flask import Flask
from flask_app.initialize import initialize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--queue", type=bool, default=False)
parser.add_argument("--cors", action='store_true', default=False)
args = parser.parse_args()

app = Flask(__name__)

initialize(app)

if __name__ == "__main__":
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
