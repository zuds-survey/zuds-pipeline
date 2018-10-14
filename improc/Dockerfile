# Using Python 3
FROM continuumio/anaconda3:latest

MAINTAINER 'Danny Goldstein <danny@caltech.edu>'

# build some key software
RUN apt-get update && apt-get install -y gcc make emacs gfortran sextractor scamp swarp  && cd

# build cfitsio
RUN wget http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio3450.tar.gz && \
    tar xvzf cfitsio3450.tar.gz && \
    cd cfitsio && \
    ./configure && make && make fpack && make funpack && \
    make install && cd - && rm cfitsio3450.tar.gz
ENV PATH ${PATH}:/cfitsio/bin


# install mpi
RUN mkdir /build/ && apt-get install -y g++

RUN cd /build && wget http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz \
  && tar xvzf mpich-3.2.tar.gz && cd /build/mpich-3.2 \
  && ./configure && make -j4 && make install && make clean && rm /build/mpich-3.2.tar.gz

RUN cd /build && wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.0.0.tar.gz \
  && tar xvzf mpi4py-3.0.0.tar.gz

RUN cd /build/mpi4py-3.0.0 && python setup.py build && python setup.py install && rm -rf /build/

RUN /sbin/ldconfig

RUN apt-get install -y wcstools

RUN wget http://cdsarc.u-strasbg.fr/ftp/pub/sw/cdsclient.tar.gz && \
    tar xvzf cdsclient.tar.gz && rm cdsclient.tar.gz && \
    cd cdsclient-3.84 && ./configure && make && make install && \
    cd -

RUN apt-get install -y plplot12-driver-cairo libcurl4-openssl-dev

RUN git clone https://github.com/acbecker/hotpants.git && \
    cd hotpants && sed -i "s/-lcfitsio/-lcfitsio -lcurl/g" Makefile && \
    make && cp hotpants /usr/bin

RUN echo 'alias sex=sextractor' >> ~/.bashrc
RUN echo 'alias swarp=SWarp' >> ~/.bashrc