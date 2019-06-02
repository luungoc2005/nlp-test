#!/bin/bash
# . botbot-env/bin/activate

# pip install nltk Flask pymagnitude scikit-learn scipy gunicorn
# pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl 
# python3 -m nltk.downloader 'punkt'

# cd ./botbot-nlp

CUDA_VISIBLE_DEVICES=-1
LOG_FILE="./flask_app/logs/main_$(date +%s)_stdout.log"
touch $LOG_FILE
gunicorn --workers=1 --timeout=500 --bind=0.0.0.0:5000 flask_app.entrypoint:app &> $LOG_FILE