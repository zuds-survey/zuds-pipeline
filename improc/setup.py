from numpy.distutils.core import setup
from numpy.distutils.core import Extension
from numpy.distutils.command.install import install
from Cython.Build import cythonize
import glob


# Create the microlens fortran extension to be compiled as a shared library using f2py
fort_sources = glob.glob('improc/subroutines/*.f90')
immath = Extension(name='improc._fsub', sources=fort_sources,
                   libraries=['cfitsio', 'curl'])

# Create the glafic c++ extension to be compiled as a shared library using cython
c_sources = glob.glob('improc/subroutines/*.cpp') + glob.glob('improc/*.pyx')

glafic = Extension(name='improc.groupimages', sources=c_sources,
                   language='c++', extra_compile_args=['-w'],
                   libraries=['cfitsio', 'm', 'curl'])

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
        glafic_module.library_dirs = library_dirs
        glafic_module.include_dirs = include_dirs + ['improc/subroutines']
        immath.library_dirs = library_dirs
        immath.include_dirs = include_dirs

glafic_module = cythonize(glafic)[0]

setup(name='improc',
      version='dev',
      packages=['improc'],
      ext_modules=[immath, glafic_module],
      package_data={'improc':['config/*']},
      cmdclass={'install':InstallCommand}
      )
