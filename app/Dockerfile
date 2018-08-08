FROM continuumio/miniconda
RUN pip install confluent-kafka
RUN conda install psycopg2
COPY broker.py /root/broker.py
CMD ["python", "broker.py"]
