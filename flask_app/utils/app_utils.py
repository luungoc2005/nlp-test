import json
import logging
from os import path
from config import CONFIG_PATH

def save_config(app):
    with open(CONFIG_PATH, 'w', encoding='utf8') as cfg_file:
        json.dump({
            'MODELS': app.config.get('MODELS', None),
        }, cfg_file)

def get_config(app):
    if path.isfile(CONFIG_PATH):
        try:
            cfg = json.load(open(CONFIG_PATH, 'r', encoding='utf8'))
            app.config.update(cfg)
        except:
            logging.warning('Failed to load configuration. Using defaults')
            pass
