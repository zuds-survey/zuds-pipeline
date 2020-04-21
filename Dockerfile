FROM continuumio/miniconda3:4.7.10

MAINTAINER Danny Goldstein <dgold@caltech.edu>

# Use bash

SHELL ["/bin/bash", "-c"]

# Install system dependencies.

RUN apt-get update \
    && apt-get install -y curl procps build-essential gfortran git subversion \
    flex pkg-config cmake \
    autoconf m4 libtool automake locales libopenblas-dev \
    && rm -fr /var/lib/apt/lists/*

# We install everything directly into /usr so that we do
# not need to modify the default library and executable
# search paths.  Shifter will manipulate LD_LIBRARY_PATH,
# so it is important not to use that variable.

# Create working directory for builds

RUN mkdir /usr/src/zuds
WORKDIR /usr/src/zuds

# The conda TCL packages overwrite the system-installed regex.h.  So
# now we force reinstall of the package that provides that

RUN apt-get update \
    && apt-get install -y --reinstall libc6-dev \
    && rm -fr /var/lib/apt/lists/*

# Copy all patch files to current working directory

RUN mkdir ./rules
ADD patches/patch_* ./rules/

# Install MPICH 3.2 which is compatible with the external
# Cray MPICH which is prepended to LD_LIBRARY_PATH as part
# of shifter.

RUN curl -SL http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz \
    -o mpich-3.2.tar.gz \
    && tar -xzf mpich-3.2.tar.gz \
    && cd mpich-3.2 \
    && CC="gcc" CXX="g++" CFLAGS="-O3 -fPIC -pthread" CXXFLAGS="-O3 -fPIC -pthread" ./configure  --prefix=/usr \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf mpich-3.2*

# Install CFITSIO.

RUN curl -SL http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio3410.tar.gz \
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

RUN curl -SL https://heasarc.gsfc.nasa.gov/docs/software/ftools/fitsverify/fitsverify-4.18.tar.gz \
    -o fitsverify-4.18.tar.gz \
    && tar xzf fitsverify-4.18.tar.gz \
    && cd fitsverify \
    && gcc -O3 -fPIC -pthread -I/usr/include -DSTANDALONE -o fitsverify ftverify.c \
    fvrf_data.c fvrf_file.c fvrf_head.c fvrf_key.c fvrf_misc.c \
    -L/usr/lib -lcfitsio -lm \
    && cp -a fitsverify "/usr/bin/" \
    && cd .. \
    && rm -rf fitsverify*


# wcstools

RUN curl -SL http://tdc-www.harvard.edu/software/wcstools/wcstools-3.9.5.tar.gz \
    -o wcstools-3.9.5.tar.gz \
    && tar xzf wcstools-3.9.5.tar.gz \
    && cd wcstools-3.9.5 \
    && chmod -R u+w . \
    && CC="gcc" CFLAGS="-O3 -fPIC -pthread" \
    CPPFLAGS="-I/usr/include" \
    LDFLAGS="-L/usr/lib" \
    && make -j 4 all \
    && cd ..

ENV PATH=$PATH:/usr/src/zuds/wcstools-3.9.5/bin

# Fftw3
RUN curl -SL http://www.fftw.org/fftw-3.3.5.tar.gz \
    -o fftw-3.3.5.tar.gz \
    && tar xzf fftw-3.3.5.tar.gz \
    && cd fftw-3.3.5 \
    && CC="gcc" CFLAGS="-O3 -fPIC -pthread" ./configure --enable-threads  --prefix="/usr" \
    && make -j 4 && make install \
    && make clean \
    && CC="gcc" CFLAGS="-O3 -fPIC -pthread" ./configure --enable-float --enable-threads  --prefix="/usr" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf fftw*

# Astromatic toolkit pieces

RUN apt-get update && apt-get install -y liblapacke-dev  liblapack3 liblapack-dev libopenblas-base libopenblas-dev  liblapack-dev emacs

RUN curl -SL http://www.astromatic.net/download/sextractor/sextractor-2.19.5.tar.gz \
    -o sextractor-2.19.5.tar.gz \
    && tar xzf sextractor-2.19.5.tar.gz \
    && cd sextractor-2.19.5 \
    && patch -p1 < ../rules/patch_sextractor \
    && chmod +x autogen.sh \
    && ./autogen.sh \
    && CC="gcc" CFLAGS="-O3 -fPIC -pthread" CPPFLAGS="-I/usr/include" ./configure --with-lapacke="-lm -lblas -llapack -llapacke" --prefix="/usr" \
    && make && make install \
    && cd .. \
    && rm -rf sextractor*

RUN curl -SL http://www.astromatic.net/download/swarp/swarp-2.38.0.tar.gz \
    -o swarp-2.38.0.tar.gz \
    && tar xzf swarp-2.38.0.tar.gz \
    && cd swarp-2.38.0 \
    && CC="gcc" CFLAGS="-O3 -fPIC -pthread" \
    ./configure  \
    --prefix="/usr" \
    --enable-threads \
    && make && make install \
    && cd .. \
    && rm -rf swarp*


# Create a fake home directory so that packages can create
# astromatic files if needed

RUN mkdir /home/zuds
RUN mkdir /home/zuds/.astropy

WORKDIR /home/zuds
ENV HOME /home/zuds

RUN git clone https://github.com/acbecker/hotpants.git && \
    cd hotpants && make -j4 CFITSIOINCDIR=/usr/include CFITSIOLIBDIR=/usr/lib && \
    cp hotpants /usr/bin && cd .. && rm -rf hotpants


# Install mpi4py.
RUN curl -SL https://files.pythonhosted.org/packages/04/f5/a615603ce4ab7f40b65dba63759455e3da610d9a155d4d4cece1d8fd6706/mpi4py-3.0.2.tar.gz \
    -o mpi4py-3.0.2.tar.gz \
    && tar xzf mpi4py-3.0.2.tar.gz \
    && cd mpi4py-3.0.2 \
    && python setup.py build --mpicc="mpicc" --mpicxx="mpicxx" \
    && python setup.py install \
    && cd .. \
    && rm -rf mpi4py*


RUN conda install postgresql ipython notebook
RUN apt-get update && apt-get install -y libbz2-dev

RUN echo 2
RUN pip install zuds

RUN curl https://portal.nersc.gov/cfs/m937/demo.tar.gz  -O demo.tar.gz && \
    mkdir ~/.data && cd ~/.data && tar -xvzf ../demo.tar.gz && \
    rm ../demo.tar.gz


RUN python -c "import astropy"
RUN python -c "import matplotlib.font_manager as fm; f = fm.FontManager"

ENTRYPOINT ["jupyter", "notebook", "--no-browser", "--port=8888", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]
