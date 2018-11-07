FROM continuumio/miniconda3
MAINTAINER Danny Goldstein <dgold@caltech.edu>

RUN conda install psycopg2 requests 
RUN pip install pika  schedule grequests ztfquery pandas numpy astropy

ADD poll.py /poll.py

