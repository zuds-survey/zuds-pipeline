import os
import numpy as np
from .shellcmd import execute
from .yao import yao_photometry_single
from astropy.io import fits


__all__ = ['solve_zeropoint']
__whatami__ = 'Zeropoint an image by calibrating to PS1.'
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'

abspath = os.path.abspath

matchquery = "SELECT id, ra, dec, {flt} " \
    "from dr1.ps1 where q3c_poly_query(ra, dec, "\
    "'{{{ra_ll}, {dec_ll}, {ra_lr}, {dec_lr}, {ra_ur}, {dec_ur}, {ra_ul}, {dec_ul}}}') "\
    " and {flt} > 0.0"


def xy2sky(im, x, y):
    """Convert the x and y coordinates on an image to RA, DEC in degrees."""
    command = 'xy2sky -d %s %d %d' % (im, x, y)
    stdout, stderr = execute(command)
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


def zpsee(image, psf, cat, cursor):
    """Compute the median zeropoint of an image or images (path/paths:
    `im_or_ims`) using the Pan-STARRS photometric database (cursor:
    `cursor`)."""

    with fits.open(image) as f:
        band = f[0].header['FILTER'] + '_median'
        nax1 = f[0].header['NAXIS1']
        nax2 = f[0].header['NAXIS2']

    ra_ll, dec_ll = xy2sky(image, 1, 1)
    ra_lr, dec_lr = xy2sky(image, nax1, 1)
    ra_ul, dec_ul = xy2sky(image, 1, nax2)
    ra_ur, dec_ur = xy2sky(image, nax1, nax2)

    cat = parse_sexcat(cat, bin=True)
    cat = cat[cat['FLAGS'] == 0]

    query_dict = {'flt':band, 'ra_ll':ra_ll, 'dec_ll':dec_ll,
                  'ra_lr':ra_lr, 'dec_lr':dec_lr, 'ra_ul':ra_ul,
                  'dec_ul':dec_ul, 'ra_ur':ra_ur, 'dec_ur':dec_ur}

    for key in query_dict:
        query_dict[key] = str(query_dict[key])
    cursor.execute(matchquery.format(**query_dict))

    result = np.array(cursor.fetchall(), dtype=[('id','<i8'),
                                                ('ra','<f8'),
                                                ('dec','<f8'),
                                                ('mag','<f8')])

    these_zps = []

    for row in cat:
        ra_c = row['X_WORLD']
        dec_c = row['Y_WORLD']
        sep = 3600. * ( np.cos(dec_c*np.pi/180.) * (ra_c - result['ra'])**2 + (dec_c - result['dec'])**2)**0.5

        match = result[sep <= 2.]

        for mps1 in match['mag']:

            # now calculate the PSF mag

            pobj = yao_photometry_single(image, psf, ra_c, dec_c)
            mag_c = -2.5 * np.log10(pobj.Fpsf) + 27.5

            zp = 27.5 + (mps1 - mag_c)
            these_zps.append(zp)

    with fits.open(psf) as f:
        seeing = f[1].header['PSF_FWHM'] * 1.013
    zp = np.median(these_zps) if len(these_zps) > 2 else 31.9999

    return zp, seeing


def solve_zeropoint(image, psf, cat):

    import psycopg2

    con = psycopg2.connect(dbname='desi', host='***REMOVED***',
                           port=5432, user='***REMOVED***', password='***REMOVED***')

    # takes a list of images and sextractor catalogs and computes seeing /
    # zeropoints for all of them

    # the sextractor catalogs are just to list detections
    # the photometry is done using PSF fitting in zpsee

    with con:
        cursor = con.cursor()
        zp, see = zpsee(image, psf, cat, cursor)
        with fits.open(image, mode='update') as f:
            f[0].header['MAGZP'] = zp
            f[0].header['SEEING'] = see
