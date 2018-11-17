FROM lensgrinder_nersc-job-submission-service
MAINTAINER Danny Goldstein <dgold@caltech.edu>

RUN pip install schedule grequests ztfquery

ADD poll.py /poll.py

ENTRYPOINT ["python", "/poll.py"]
