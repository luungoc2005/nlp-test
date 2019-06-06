# FROM luungoc2005/botbot-nlp:staging
FROM ubuntu
MAINTAINER Ngoc Nguyen <ngoc.nguyen@2359media.com>

# Uncomment these lines if expanding from vanilla ubuntu
RUN apt-get update
RUN apt-get install -y software-properties-common build-essential curl unzip git locales
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update
RUN apt-get install -y python3.6 python3-pip
RUN python3 -m pip install virtualenv

RUN virtualenv botbot-env 

ENV VIRTUAL_ENV=/botbot-env
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set Locale
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

RUN git clone https://github.com/facebookresearch/fastText.git
WORKDIR /fastText
RUN pip install .

WORKDIR /

ENV PYTHONIOENCODING utf8

# Can put these inside requirements.txt but... for a minimal version

COPY . /botbot-nlp
WORKDIR /botbot-nlp

RUN pip install -r requirements.txt

# RUN chmod u+x ./data/get_data_minimal.bash
# RUN ./data/get_data_minimal.bash

EXPOSE 80

# RUN chmod u+x botbot-nlp/entrypoint.sh
# CMD botbot-nlp/entrypoint.sh

# RUN mkdir -p ./flask_app/logs
# ENV LOG_FILE ./flask_app/logs/main_$(date +%s)_stdout.log
# RUN touch $LOG_FILE

# $PORT is heroku's provided port to bind to
# gunicorn --workers=1 --timeout=500 --bind=0.0.0.0:$PORT flask_app.entrypoint:app &> $LOG_FILE

ENTRYPOINT gunicorn \
  --workers=1 \
  --timeout=500 \
  --bind=0.0.0.0:${PORT:-80} \
  --log-level=error \
  flask_app.entrypoint:app
