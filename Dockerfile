FROM luungoc2005/botbot
# FROM ubuntu
MAINTAINER Ngoc Nguyen <luungoc2005@2359media.com>

# Uncomment these lines if expanding from vanilla ubuntu
# RUN add-apt-repository ppa:jonathonf/python-3.6
# RUN apt-get update
# RUN apt-get install -y build-essential curl unzip python3.6

# Can put these inside requirements.txt but... for a minimal version
RUN pip install tensorflow tensorboardX tqdm nltk pytorch Cython Flask

COPY . /botbot_nlu
WORKDIR /botbot_nlu

RUN chmod u+x ./data/get_data_minimal.bash
RUN ./data/get_data_minimal.bash
RUN python setup.py build_ext --inplace

EXPOSE 5000
ENTRYPOINT gunicorn -w 1 -t 1000 -b 127.0.0.1:5000 start_flask:app