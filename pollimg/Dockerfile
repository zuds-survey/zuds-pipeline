FROM continuumio/miniconda3
MAINTAINER Danny Goldstein <dgold@caltech.edu>

RUN pip install Celery

ADD poll.py /poll.py
ENTRYPOINT ['python', '/poll.py']
