#!/bin/bash

# build cfitsio
curl -SL http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio3410.tar.gz \
    -o cfitsio3410.tar.gz \
    && tar xzf cfitsio3410.tar.gz \
    && cd cfitsio \
    && CC="gcc" CFLAGS="-O3 -fPIC -pthread" ./configure  \
    --prefix=/usr --enable-reentrant \
    && make stand_alone \
    && make utils \
    && make shared \
    && make install \
    && cd .. \
    && rm -rf cfitsio*

# terapix software
conda install -c conda-forge astromatic-swarp astromatic-source-extractor

# hotpants
git clone https://github.com/acbecker/hotpants.git \
    && cd hotpants \
    && sed -e -i 's/#include<malloc.h>//g' *.c
    && make
    && make install

# non-module binaries
conda install postgresql ipython notebook bzip2

# install zuds
pip install zuds


