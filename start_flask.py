from flask import Flask
from flask_app.initialize import initialize

app = Flask(__name__)

initialize(app)

if __name__ == "__main__":
    app.run(processes=1, debug=True)
