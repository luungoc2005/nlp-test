FROM luungoc2005/botbot-nlp:staging
# FROM ubuntu
MAINTAINER Ngoc Nguyen <ngoc.nguyen@2359media.com>

# Uncomment these lines if expanding from vanilla ubuntu
# RUN apt-get update
# RUN apt-get install -y software-properties-common build-essential curl unzip
# RUN add-apt-repository ppa:jonathonf/python-3.6
# RUN apt-get update
# RUN apt-get install -y python3.6 python3-pip
# RUN python3 -m pip install virtualenv

# RUN virtualenv botbot-env 

# Can put these inside requirements.txt but... for a minimal version

COPY . /botbot-nlp

# RUN chmod u+x ./data/get_data_minimal.bash
# RUN ./data/get_data_minimal.bash

EXPOSE 5000

# Flags:
# -w: number of workers
# -t: timeout for each request (in seconds)
RUN chmod u+x botbot-nlp/entrypoint.sh
ENTRYPOINT botbot-nlp/entrypoint.sh