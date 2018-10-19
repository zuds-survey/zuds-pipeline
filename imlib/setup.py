from numpy.distutils.core import setup
from numpy.distutils.core import Extension
from numpy.distutils.command.install import install
import glob
from Cython.Build import cythonize


# Create the fortran extension to be compiled as a shared library using f2py
fort_sources = glob.glob('imlib/fits/*.f90')
ffits = Extension(name='fits._ffits', sources=fort_sources, libraries=['cfitsio', 'curl'],
                  extra_compile_args=['-w', '-O3'], extra_f90_compile_args=['-w', '-O3'])

# Create the C++ extension to be compiled as a shared library using cython
c_sources = glob.glob('imlib/fits/*.pyx') + glob.glob('imlib/fits/*.cc')
cfits = Extension(name='fits._cfits', sources=c_sources, libraries=['cfitsio', 'curl'],
                  extra_compile_args=['-w'], language='c++')
cfmod = cythonize(cfits)[0]


class InstallCommand(install):
    user_options = install.user_options + [
        ('cfitsio-inc=', None, 'cfitsio header search path.'),
        ('cfitsio-libdir=', None, 'cfitsio library search path.'),
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.cfitsio_inc = None
        self.cfitsio_libdir = None

    def finalize_options(self):
        install.finalize_options(self)
        library_dirs = [d for d in [self.cfitsio_libdir] if d]
        include_dirs = [d for d in [self.cfitsio_inc] if d]
        ffits.library_dirs = library_dirs
        ffits.include_dirs = include_dirs
        cfmod.library_dirs = library_dirs
        cfmod.include_dirs = include_dirs + ['imlib/fits/']


setup(name='imlib',
      ext_package='imlib',
      packages=['imlib', 'imlib.proc'],
      version='dev',
      ext_modules=[ffits, cfmod],
      cmdclass={'install':InstallCommand}
      )
