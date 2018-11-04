FROM continuumio/miniconda3
MAINTAINER Danny Goldstein <dgold@caltech.edu>

RUN pip install pika astropy
RUN conda install psycopg2

ADD queuemon.py /queuemon.py

