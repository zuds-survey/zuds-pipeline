FROM pipeline
MAINTAINER Danny Goldstein <dgold@caltech.edu>

RUN conda install requests
RUN pip install pika  schedule grequests ztfquery

ADD poll.py /poll.py

CMD ["python", "/poll.py"]

