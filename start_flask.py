from flask import Flask
from flask_app.initialize import initialize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--queue", type=bool, default=False)
args = parser.parse_known_args()

app = Flask(__name__)

initialize(app)

if __name__ == "__main__":
    if args.debug:
        app.run(processes=1, debug=True, threaded=False)
    else:
        app.config['USE_QUEUE'] = args.queue
        app.run()
