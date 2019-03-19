from flask_app.entrypoint import app

from flask_app.utils import get_json, jsonerror
from flask_app.utils.app_utils import get_config, save_config
from os import path, remove
import sys, traceback

@app.route("/delete", methods=['POST'])
def flask_delete_model():
    try:
        content = get_json(request)

        if 'model_id' not in content:
            return jsonerror('Missing model_id argument in request')
        else:
            model_id = content['model_id']

            get_config(app)

            all_models = app.config['MODELS']

            if model_id not in all_models:
                return jsonerror('Model ID not found')
            else:
                model_config = all_models[model_id]

                if path.exists(model_config['CLF_MODEL_PATH']):
                    remove(model_config['CLF_MODEL_PATH'])
                
                if path.exists(model_config['ENT_MODEL_PATH']):
                    remove(model_config['ENT_MODEL_PATH'])

                del all_models[model_id]

                app.config['MODELS'] = all_models

                save_config(app)

    except:
        traceback.print_exc(limit=2, file=sys.stdout)
        return jsonerror('Runtime exception encountered when handling request')
