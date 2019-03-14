from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u

import numpy as np
import subprocess
import os

abspath = os.path.abspath

matchquery = "SELECT id, ra, dec, {flt} " \
    "from dr1.ps1 where q3c_poly_query(ra, dec, "\
    "'{{{ra_ll}, {dec_ll}, {ra_lr}, {dec_lr}, {ra_ur}, {dec_ur}, {ra_ul}, {dec_ul}}}') "\
    " and {flt} > 0.0"

def parse_sexcat(cat, bin=False):
    """Read a sextractor catalog file (path: `cat`) and return a numpy
    record array containing the values."""

    if bin:
        data = fits.open(cat)[2].data
    else:
        data = np.genfromtxt(cat, dtype=None)
    return data


def zpsee(image, cat, cursor, zp_fid, inhdr=None):
    """Compute the median zeropoint of an image or images (path/paths:
    `im_or_ims`) using the Pan-STARRS photometric database (cursor:
    `cursor`)."""

    with fits.open(image) as f:
        band = f[0].header['FILTER'] + '_median'
        nax1 = f[0].header['NAXIS1']
        nax2 = f[0].header['NAXIS2']
        hd = f[0].header.copy()

    if inhdr is not None:
        with open(inhdr, 'r') as f:
            for line in f:
                card = fits.Card.fromstring(f.strip())
                hd.append(card)

    wcs = WCS(hd)
    corners = wcs.calc_footprint()

    ra_ll, dec_ll = corners[0]
    ra_lr, dec_lr = corners[1]
    ra_ul, dec_ul = corners[2]
    ra_ur, dec_ur = corners[3]

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

    cat_coords = SkyCoord(ra=cat['X_WORLD'] * u.degree, dec=cat['Y_WORLD'] * u.degree)
    db_coords = SkyCoord(ra=result['ra'] * u.degree, dec=result['dec'] * u.degree)
    idx, d2d, _ = cat_coords.match_to_catalog_sky(db_coords)

    for cat_row, i, d in zip(cat, idx, d2d):
        if d <= 1 * u.arcsec:
            ps1_mag = result[i]['mag']
            sex_mag = cat_row['MAG_PSF']
            zp = zp_fid + ps1_mag - sex_mag
            seeing = cat_row['FWHM_IMAGE'] 
            zps.append(zp)

    return np.median(zps)


def solve_zeropoint(image, cat, psf, zp_fid=27.5):

    import psycopg2

    con = psycopg2.connect(dbname='desi', host='***REMOVED***',
                           port=5432, user='***REMOVED***', password='***REMOVED***')

    # takes an image and sextractor  catalog and computes seeing /
    # zeropoints for all of them

    with con:
        cursor = con.cursor()
        inhdr = image.replace('.fits', '.head')
        zp = zpsee(image, cat, cursor, zp_fid, inhdr=inhdr)
        
    with fits.open(psf) as f, fits.open(image) as im:
        seeing = f[1].header['PSF_FWHM'] * im[0].header['PIXSCALE']
        
    with fits.open(image, mode='update') as f:
        f[0].header['MAGZP'] = zp
        f[0].header['SEEING'] = seeing
    
    

def calibrate(frame):
    # astrometrically and photometrically calibrate a ZTF frame using
    # PSF fitting and gaia

    mypath = os.path.abspath(__file__)
    sexconf = os.path.join(mypath, '../astromatic/calibration/scamp.sex')
    sexparam = os.path.join(mypath, '../astromatic/calibration/scamp.param')
    scampconf = os.path.join(mypath, '../astromatic/calibration/scamp.conf')
    sexphotconf = os.path.join(mypath, '../astromatic/calibration/psfphot.sex')
    sexnnw = os.path.join(mypath, '../astromatic/calibration/default.nnw')
    sexfilter = os.path.join(mypath, '../astromatic/calibration/default.conv')
    nnwfilt = f'-FILTER_NAME {sexfilter} -STARNNW_NAME {sexnnw}'

    # first run source extractor
    cat = frame.replace('.fits', '.cat')
    cmd = f'sex -c {sexconf} -PARAMETERS_NAME {sexparam} -CATALOG_NAME {cat} {nwwfilt} {frame}'
    subprocess.check_call(cmd.split())

    # now run scamp to solve the astrometry
    cmd = f'scamp -c {scampconf} {cat}'
    subprocess.check_call(cmd.split())

    # now get a model of the psf
    cmd = f'psfex -c {psfconf} {cat}'
    subprocess.check_call(cmd.split())

    # now do photometry by fitting the psf model to the image
    psf = frame.replace('.fits', '.psf')
    cmd = f'sex -c {sexphotconf} -CATALOG_NAME {cat} -PSF_NAME {psf} -PARAMETERS_NAME {photparams} {nnwfilt} {frame}'
    subprocess.check_call(cmd.split())

    # now solve for the zeropoint
    solve_zeropoint(frame, cat, psf, zp_fid=27.5)
