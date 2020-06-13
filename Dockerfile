FROM continuumio/miniconda3:4.7.10

MAINTAINER Danny Goldstein <dgold@caltech.edu>

# Install astromatic software
RUN conda install -c conda-forge astromatic-swarp astromatic-source-extractor

# Install hotpants
RUN apt-get update && apt-get install -y libcfitsio-dev libcurl4-openssl-dev postgresql-client postgresql \
    libpq-dev make gcc libbz2-dev curl && \
    git clone https://github.com/zuds-survey/hotpants.git && \
    cd hotpants && LIBS="-lm -lcfitsio -lcurl" make -e && \
    cp hotpants $CONDA_PREFIX/bin && cd -


SHELL ["/bin/bash", "-c"]

ADD setup.py zuds requirements.txt zuds-pipeline/

# move TMPDIR off /tmp which is small on docker
#RUN df -h && mkdir $HOME/.piptemp $HOME/.pipbuild $HOME/.pipcache && \
#    TMPDIR=$HOME/.piptemp pip install --cache-dir=$HOME/.pipcache --build
#$HOME/.pipbuild . jupyter && \
#    rm -r $HOME/.piptemp $HOME/.pipbuild $HOME/.pipcache

RUN cd zuds-pipeline && pip install jupyter && \
    pip install -r requirements.txt && \
    pip install . && cd -


RUN curl https://portal.nersc.gov/cfs/m937/demo.tar.gz -o demo.tar.gz && \
    mkdir ~/.data && cd ~/.data && tar -xvzf ../demo.tar.gz && \
    rm ../demo.tar.gz

ENTRYPOINT ["jupyter", "notebook", "--no-browser", "--port=8888", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]
