# FROM luungoc2005/botbot
FROM ubuntu
MAINTAINER Ngoc Nguyen <luungoc2005@2359media.com>

# Uncomment these lines if expanding from vanilla ubuntu
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update
RUN apt-get install -y build-essential curl unzip python3.6

# Can put these inside requirements.txt but... for a minimal version
RUN pip3 install nltk Flask pymagnitude scikit-learn gunicorn
RUN pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl 

COPY . /botbot_nlu
WORKDIR /botbot_nlu

RUN chmod u+x ./data/get_data_minimal.bash
RUN ./data/get_data_minimal.bash

EXPOSE 5000
ENTRYPOINT gunicorn -w 2 -t 1000 -b 127.0.0.1:5000 start_flask:app