import os
import subprocess
import numpy as np

__all__ = ['solve_zeropoint']
__whatami__ = 'Zeropoint an image by calibrating to PS1.'
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'

abspath = os.path.abspath

matchquery = "SELECT id, ra, dec, {flt} " \
    "from dr1.ps1 where q3c_poly_query(ra, dec, "\
    "'{{{ra_ll}, {dec_ll}, {ra_lr}, {dec_lr}, {ra_ur}, {dec_ur}, {ra_ul}, {dec_ul}}}') "\
    " and {flt} > 0.0"


def _append_config(command, config):
    config = abspath(config)
    command += (' -c %s' % config)
    return command


def _coerce(path_or_paths):
    return ' '.join(np.atleast_1d(path_or_paths).tolist())


def _execute(cmd):
    """Execute a shell command, log the stdout and stderr, and check
    the return code. If the return code is != 0, raise an
    exception."""


    args = cmd.split()
    process = subprocess.Popen( args, stdout = subprocess.PIPE, stderr = subprocess.PIPE )
    stdout, stderr = process.communicate()

    # check return code
    if process.returncode != 0:
        raise Exception(stderr)

    return stdout, stderr


def pathsplit(path):
    """Return the absolute root of `path`, and the filename of
    `path`."""

    apath = abspath(path)
    spath = apath.split('/')

    root = '/'.join(spath[:-1])
    fname = spath[-1]

    return (root, fname)


def xy2sky(im, x, y):
    """Convert the x and y coordinates on an image to RA, DEC in degrees."""
    command = 'xy2sky -d %s %d %d' % (im, x, y)
    stdout, stderr = _execute(command)
    ra, dec, epoch, x, y = stdout.split()
    return float(ra), float(dec)


def parse_sexcat(cat, bin=False):
    """Read a sextractor catalog file (path: `cat`) and return a numpy
    record array containing the values."""

    if bin:
        data = fits.open(cat)[2].data
    else:
        data = np.genfromtxt(cat, dtype=None)
    return data


def parse_sexconf(conf):
    """Read a sextractor configuration file (path: `conf`) and return
    the contents as a dictionary."""
    d = {}
    with open(conf, 'r') as f:
        for line in f:
            d.update(dict([line.split()]))
    return d


def sex(im, config=None):
    """Run sextractor on an image (path: `im`), using a config file
    (path: `config`).  If `config` is None, use the default
    config."""

    im = abspath(im)
    command = "sex %s" % im

    if config is not None:
        command = _append_config(command, config)
    _execute(command)


def swarp(im_or_ims, config=None):
    """Run swarp on an image or list of images (path/paths:
    `im_or_ims`), using a config file (path: `config`). If `config` is
    None, use the default config."""

    targetstr = _coerce(im_or_ims)
    command = "swarp %s" % targetstr

    if config is not None:
        command = _append_config(command, config)
    _execute(command)


def scamp(cat_or_cats, config=None):
    """Run scamp on a catalog or list of catalogs (path/paths:
    `cat_or_cats`), using a config file (path: `config`). If `config` is
    None, use the default config."""

    targetstr = _coerce(cat_or_cats)
    command = "scamp %s" % targetstr

    if config is not None:
        command = _append_config(command, config)
    _execute(command)


def zpsee(im_or_ims, cat_or_cats, cursor):
    """Compute the median zeropoint of an image or images (path/paths:
    `im_or_ims`) using the Pan-STARRS photometric database (cursor:
    `cursor`)."""

    from astropy.io import fits


    paths = np.atleast_1d(im_or_ims).tolist()
    catpaths = np.atleast_1d(cat_or_cats).tolist()

    hdulists = [fits.open(path) for path in paths]
    filters = [hdul[0].header['FILTER'] + '_median' for hdul in hdulists]
    nax1 = [hdul[0].header['NAXIS1'] for hdul in hdulists]
    nax2 = [hdul[0].header['NAXIS2'] for hdul in hdulists]

    ra_ll, dec_ll = np.array([xy2sky(im, 1, 1) for i,im in enumerate(paths)]).T
    ra_lr, dec_lr = np.array([xy2sky(im, nax1[i], 1) for i,im in enumerate(paths)]).T
    ra_ul, dec_ul = np.array([xy2sky(im, 1, nax2[i]) for i,im in enumerate(paths)]).T
    ra_ur, dec_ur = np.array([xy2sky(im, nax1[i], nax2[i]) for i,im in enumerate(paths)]).T

    zps = []
    seeings = []

    for i,im in enumerate(paths):
        cat = parse_sexcat(catpaths[i], bin=True)
        cat = cat[cat['FLAGS'] == 0]
        these_zps = []
        these_seeings = []

        query_dict = {'flt':filters[i], 'ra_ll':ra_ll[i], 'dec_ll':dec_ll[i],
                      'ra_lr':ra_lr[i], 'dec_lr':dec_lr[i], 'ra_ul':ra_ul[i],
                      'dec_ul':dec_ul[i], 'ra_ur':ra_ur[i], 'dec_ur':dec_ur[i]}

        for key in query_dict:
            query_dict[key] = str(query_dict[key])
        cursor.execute(matchquery.format(**query_dict))

        result = np.array(cursor.fetchall(), dtype=[('id','<i8'),
                                                    ('ra','<f8'),
                                                    ('dec','<f8'),
                                                    ('mag','<f8')])
        for row in cat:
            mag_c = row['MAG_AUTO']
            ra_c = row['X_WORLD']
            dec_c = row['Y_WORLD']
            fwhm = row['FWHM_IMAGE']

            sep = 3600. * ( np.cos(dec_c*np.pi/180.) * (ra_c - result['ra'])**2 + (dec_c - result['dec'])**2)**0.5

            match = result[sep <= 2.]
            seeing = fwhm * 1.01 # arcsec (1.01 = PTF plate scale)
            these_seeings.append(seeing)


            for mps1 in match['mag']:
                zp = 27.5 + (mps1 - mag_c)
                these_zps.append(zp)

        seeings.append(np.median(these_seeings) if len(these_seeings) > 2 else 1.9999)
        zps.append(np.median(these_zps) if len(these_zps) > 2 else 31.9999)

    return np.array(zps), np.array(seeings)


def solve_zeropoint(image, cat):

    import psycopg2

    con = psycopg2.connect(dbname='desi', host='***REMOVED***',
                           port=5432, user='***REMOVED***', password='***REMOVED***')

    # takes a list of images and sextractor catalogs and computes seeing /
    # zeropoints for all of them

    images = np.atleast_1d(np.genfromtxt(image, dtype=str)).tolist()
    cats = np.atleast_1d(np.genfromtxt(cat, dtype=str)).tolist()

    with con:
        cursor = con.cursor()
        zp, see = zpsee(images, cats, cursor)
        for zeropoint, seeing, image in zip(zp, see, images):
            with fits.open(image, mode='update') as f:
                f[0].header['MAGZP'] = zeropoint
                f[0].header['SEEING'] = seeing
