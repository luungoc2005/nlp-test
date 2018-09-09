FROM luungoc2005/botbot-nlp:staging
# FROM ubuntu
MAINTAINER Ngoc Nguyen <luungoc2005@2359media.com>

# Uncomment these lines if expanding from vanilla ubuntu
# RUN apt-get update
# RUN apt-get install -y software-properties-common build-essential curl unzip
# RUN add-apt-repository ppa:jonathonf/python-3.6
# RUN apt-get update
# RUN apt-get install -y python3.6 python3-pip
# RUN python3 -m pip install virtualenv

# RUN virtualenv botbot-env 

RUN source botbot-env/bin/activate
# Can put these inside requirements.txt but... for a minimal version
# RUN pip install nltk Flask pymagnitude scikit-learn scipy gunicorn
# RUN pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl 
# RUN python3 -m nltk.downloader 'punkt'

COPY . /botbot-nlp
WORKDIR /botbot-nlp

# RUN chmod u+x ./data/get_data_minimal.bash
# RUN ./data/get_data_minimal.bash

EXPOSE 5000

# Flags:
# -w: number of workers
# -t: timeout for each request (in seconds)
ENTRYPOINT gunicorn -w 2 -t 500 -b 0.0.0.0:5000 start_flask:app