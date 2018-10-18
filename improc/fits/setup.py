from numpy.distutils.core import setup
from numpy.distutils.core import Extension
from numpy.distutils.command.install import install
import glob
from Cython.Build import cythonize


# Create the fortran extension to be compiled as a shared library using f2py
fort_sources = glob.glob('fits/*.f90')
ffits = Extension(name='fits._ffits', sources=fort_sources, libraries=['cfitsio', 'curl'],
                  extra_compile_args=['-w'], extra_f90_compile_args=['-w'])

c_sources = glob.glob('fits/*.pyx') + glob.glob('fits/*.cc')
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
        cfmod.include_dirs = include_dirs + ['fits/']


setup(name='fits',
      packages=['fits'],
      version='dev',
      ext_modules=[ffits, cfmod],
      package_data={'improc':['config/*']},
      cmdclass={'install':InstallCommand}
      )
