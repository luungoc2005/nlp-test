from flask_app.entrypoint import app

from config import BASE_PATH

@app.route("/")
def index():
    return "Server is up and running! BASE_PATH: %s" % BASE_PATH
