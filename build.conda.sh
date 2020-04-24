#!/bin/bash

# this script installs everything but hotpants, which needs to be
# downloaded and compiled manually. for more information on how to
# install hotpants, see README.md

# terapix software and cfitsio
conda install -c conda-forge astromatic-swarp astromatic-source-extractor cfitsio

# non-module binaries
conda install postgresql ipython notebook bzip2

# install zuds
pip install zuds
