FROM pipeline
MAINTAINER Danny Goldstein <dgold@caltech.edu>

RUN pip install pika

ADD queuemon.py /queuemon.py

CMD ["python", "/queuemon.py"]

