FROM lensgrinder_nersc-job-submission-service
MAINTAINER Danny Goldstein <dgold@caltech.edu>

RUN pip install ztfquery # schedule

ADD poll.py /poll.py

ENTRYPOINT ["python", "/poll.py"]
#ENTRYPOINT ["ipython"]
