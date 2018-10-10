. botbot-env/bin/activate

# pip install nltk Flask pymagnitude scikit-learn scipy gunicorn
# pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl 
# python3 -m nltk.downloader 'punkt'

cd ./botbot-nlp
gunicorn -w 1 -t 500 -b 127.0.0.1:5000 start_flask:app