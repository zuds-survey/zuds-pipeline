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

groupimages: groupimages.o fits/gethead.o
	$(LINKCC) -o $@ $^ -L$(LIBDIR1) -L$(LIBDIR2) -L$(CFITSLINK) -l$(PYLIB) -lcfitsio $(LIBS) $(SYSLIBS) $(LINKFORSHARED) -lcurl

groupimages.o: groupimages.cpp
	$(CC) -c $^ -I$(INCDIR) -I$(PLATINCDIR) -I$(CFITSDIR)

fits/gethead.o: fits/gethead.cc
	$(CC) -o $@ -c $^ -I$(INCDIR) -I$(PLATINCDIR) -I$(CFITSDIR)

groupimages.cpp: groupimages.pyx
	@cython --embed --cplus groupimages.pyx

all: groupimages

clean:
	@echo Cleaning Demos/embed
	@rm -f *~ *.o *.so core core.* *.c groupimages test.output

test: clean all
	PYTHONHOME=$(PYPREFIX) LD_LIBRARY_PATH=$(LIBDIR1):$$LD_LIBRARY_PATH ./groupimages > test.output
	$(PYTHON) assert_equal.py groupimages.output test.output
