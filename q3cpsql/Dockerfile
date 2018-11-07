FROM postgres:latest

MAINTAINER "Danny Goldstein <danny@caltech.edu>"

RUN apt-get update && apt-get install -y postgresql-q3c

ADD ./init-user-db.sh /docker-entrypoint-initdb.d/init-user-db.sh
