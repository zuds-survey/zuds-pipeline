FROM lensgrinder_nersc-job-submission-service
MAINTAINER Danny Goldstein <dgold@caltech.edu>

RUN pip install pika

ADD queuemon.py /queuemon.py

ENTRYPOINT ["python", "/queuemon.py"]

