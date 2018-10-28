FROM continuumio/miniconda3
MAINTAINER Danny Goldstein <dgold@caltech.edu>

RUN apt-get update && apt-get install rabbitmq-server
RUN pip install Celery

ADD start-server.sh start-server.sh
ENTRYPOINT ['/bin/bash', 'start-server.sh']
