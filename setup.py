from setuptools import setup

import locale
import os
import subprocess
import warnings


def _decode_stdio(stream):
    try:
        stdio_encoding = locale.getdefaultlocale()[1] or 'utf-8'
    except ValueError:
        stdio_encoding = 'utf-8'

    try:
        text = stream.decode(stdio_encoding)
    except UnicodeDecodeError:
        # Final fallback
        text = stream.decode('latin1')

    return text


def get_git_devstr(sha=False, show_warning=True, path=None):
    """
    Determines the number of revisions in this repository.
    Parameters
    ----------
    sha : bool
        If True, the full SHA1 hash will be returned. Otherwise, the total
        count of commits in the repository will be used as a "revision
        number".
    show_warning : bool
        If True, issue a warning if git returns an error code, otherwise errors
        pass silently.
    path : str or None
        If a string, specifies the directory to look in to find the git
        repository.  If `None`, the current working directory is used, and must
        be the root of the git repository.
        If given a filename it uses the directory containing that file.
    Returns
    -------
    devversion : str
        Either a string with the revision number (if `sha` is False), the
        SHA1 hash of the current commit (if `sha` is True), or an empty string
        if git version info could not be identified.
    """

    if path is None:
        path = os.getcwd()

    if not os.path.isdir(path):
        path = os.path.abspath(os.path.dirname(path))

    if sha:
        # Faster for getting just the hash of HEAD
        cmd = ['rev-parse', 'HEAD']
    else:
        cmd = ['rev-list', '--count', 'HEAD']

    def run_git(cmd):
        try:
            p = subprocess.Popen(['git'] + cmd, cwd=path,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 stdin=subprocess.PIPE)
            stdout, stderr = p.communicate()
        except OSError as e:
            if show_warning:
                warnings.warn('Error running git: ' + str(e))
            return (None, b'', b'')

        if p.returncode == 128:
            if show_warning:
                warnings.warn('No git repository present at {0!r}! Using '
                              'default dev version.'.format(path))
            return (p.returncode, b'', b'')
        if p.returncode == 129:
            if show_warning:
                warnings.warn('Your git looks old (does it support {0}?); '
                              'consider upgrading to v1.7.2 or '
                              'later.'.format(cmd[0]))
            return (p.returncode, stdout, stderr)
        elif p.returncode != 0:
            if show_warning:
                warnings.warn('Git failed while determining revision '
                              'count: {0}'.format(_decode_stdio(stderr)))
            return (p.returncode, stdout, stderr)

        return p.returncode, stdout, stderr

    returncode, stdout, stderr = run_git(cmd)

    if not sha and returncode == 128:
        # git returns 128 if the command is not run from within a git
        # repository tree. In this case, a warning is produced above but we
        # return the default dev version of '0'.
        return '0'
    elif not sha and returncode == 129:
        # git returns 129 if a command option failed to parse; in
        # particular this could happen in git versions older than 1.7.2
        # where the --count option is not supported
        # Also use --abbrev-commit and --abbrev=0 to display the minimum
        # number of characters needed per-commit (rather than the full hash)
        cmd = ['rev-list', '--abbrev-commit', '--abbrev=0', 'HEAD']
        returncode, stdout, stderr = run_git(cmd)
        # Fall back on the old method of getting all revisions and counting
        # the lines
        if returncode == 0:
            return str(stdout.count(b'\n'))
        else:
            return ''
    elif sha:
        return _decode_stdio(stdout)[:40]
    else:
        return _decode_stdio(stdout).strip()


# Need a recursive glob to find all package data files if there are
# subdirectories
import fnmatch


def recursive_glob(basedir, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(basedir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches


# Set affiliated package-specific settings
PACKAGENAME = 'zuds'
DESCRIPTION = 'ZTF/ZUDS image processing pipeline'
LONG_DESCRIPTION = ''
AUTHOR = 'The ZUDS Developers'
AUTHOR_EMAIL = 'danny@caltech.edu'
LICENSE = 'BSD'
URL = 'https://github.com/zuds-survey/zuds-pipeline'

# VERSION should be PEP386 compatible (http://www.python.org/dev/peps/pep-0386)
VERSION = '0.1.dev'

# Indicates if this version is a release version
RELEASE = 'dev' not in VERSION

if not RELEASE:
    VERSION += get_git_devstr(sha=False)

setup(name=PACKAGENAME,
      version=VERSION,
      description=DESCRIPTION,
      packages=['zuds', 'baselayer',
                'baselayer.app',
                'baselayer.app.handlers',
                'skyportal', 'skyportal.tests',
                'skyportal.tests.tools',
                'skyportal.tests.frontend',
                'skyportal.utils',
                'skyportal.handlers',
                'skyportal.handlers.api',
                'skyportal.handlers.api.internal'],
      package_dir={'zuds': 'zuds',
                   'skyportal': 'zuds/skyportal/skyportal',
                   'baselayer': 'zuds/skyportal/baselayer'},
      package_data={'zuds': ['alert_schemas/*', 'astromatic/*',
                             'config/*', 'astromatic/makecoadd/*',
                             'astromatic/makesub/*']},
      install_requires=['astropy>=4',
                        'avro-python3>=1.8.2',
                        'confluent-kafka>=1.4.0',
                        'fastavro>=0.21.7',
                        'fitsio>=1.0.5',
                        'matplotlib>=3.1.1',
                        'numpy>=1.16.4',
                        'pandas>=0.24.2',
                        'penquins>=1.0.3',
                        'photutils>=0.7.1',
                        'psycopg2>=2.8.3',
                        'PyYAML>=5.1.1',
                        'requests>=2.22.0',
                        'scipy>=1.3.0',
                        'SQLAlchemy>=1.3.7',
                        'tensorflow>=2.0.0',
                        'ztfquery>=1.6.2',
                        "supervisor>=4",
                        "psycopg2-binary",
                        "pyyaml",
                        "tornado==6.0.3",
                        "pyzmq>=0.17",
                        "pyjwt",
                        "distributed>=1.21.1",
                        "simplejson",
                        "requests",
                        "selenium>=3.12.0",
                        "pytest==4.3.1",
                        "social-auth-core>=3.0.0",
                        "social-auth-app-tornado>=1.0.0",
                        "social-auth-storage-sqlalchemy>=1.1.0",
                        "selenium-requests",
                        "arrow==0.15.4",
                        "jinja2",
                        "python-dateutil",
                        'dask>=0.15.0',
                        'joblib>=0.11',
                        'bokeh==0.12.9',
                        'pytest-randomly>=2.1.1',
                        'factory-boy==2.11.1',
                        'tqdm>=4.23.2',
                        'astroquery>=0.4',
                        'sqlalchemy-utils',
                        'apispec>=3.2.0',
                        'marshmallow>=3.4.0',
                        'marshmallow-sqlalchemy>=0.21.0',
                        'marshmallow-enum>=1.5.1',
                        'Pillow>=6',
                        'sncosmo'],
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url=URL,
      long_description=LONG_DESCRIPTION,
)

