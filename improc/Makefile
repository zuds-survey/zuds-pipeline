# Makefile for creating our standalone Cython program
PYTHON := python
PYVERSION := $(shell $(PYTHON) -c "import sys; print(sys.version[:3])")
PYPREFIX := $(shell $(PYTHON) -c "import sys; print(sys.prefix)")

INCDIR := $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_python_inc())") 
CFITSDIR := /cfitsio/include
CFITSLINK := /cfitsio/lib
PLATINCDIR := $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_python_inc(plat_specific=True))")
LIBDIR1 := $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
LIBDIR2 := $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_config_var('LIBPL'))")
PYLIB := $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_config_var('LIBRARY')[3:-2])")

CC := g++
LINKCC := g++
LINKFORSHARED := $(shell $(PYTHON) -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('LINKFORSHARED'))")
LIBS := $(shell $(PYTHON) -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('LIBS'))")
SYSLIBS :=  $(shell $(PYTHON) -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('SYSLIBS'))")
BUILDDIR := build

groupimages: $(BUILDDIR)/groupimages.o $(BUILDDIR)/gethead.o
	$(LINKCC)  -fPIC -o $@ $^ -Wl,--rpath=/opt/conda/lib -L$(LIBDIR1) -L$(LIBDIR2) -L$(CFITSLINK) -l$(PYLIB) -lcfitsio $(LIBS) $(SYSLIBS) $(LINKFORSHARED) -lcurl

$(BUILDDIR)/groupimages.o: $(BUILDDIR)/groupimages.cpp
	$(CC) -o $@ -c $^ -I$(INCDIR) -I$(PLATINCDIR) -I$(CFITSDIR) -I./fits/

$(BUILDDIR)/gethead.o: fits/gethead.cc
	$(CC) -o $@ -c $^ -I$(INCDIR) -I$(PLATINCDIR) -I$(CFITSDIR) -I./fits/

$(BUILDDIR)/groupimages.cpp: groupimages.pyx
	@mkdir -p $(BUILDDIR)
	@cython --embed --cplus groupimages.pyx -o $@

all: groupimages

clean:
	@rm -rf $(BUILDDIR) groupimages
