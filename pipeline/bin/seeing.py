import db
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from secrets import get_secret
import psycopg2


def private_logon():
    """ Log on to the ztfimages database on private """
    conn = psycopg2.connect(
            host=get_secret('hpss_dbhost'), user=get_secret('hpss_dbusername'),
            password=get_secret('hpss_dbpassword'),
            database=get_secret('olddb'),
            port=get_secret('hpss_dbport'))
    cur = conn.cursor()
    return cur


def estimate_seeing(image):
    """Estimate the seeing on an image by comparing its catalog to GAIA stars."""

    # connect to kowalski to query gaia

    catalog = image.catalog
    cursor = private_logon()

    if catalog is None or not catalog.ismapped:
        catalog = db.PipelineFITSCatalog.from_image(image)

    q = f'''SELECT "RA", "DEC" FROM dr8_north WHERE
    q3c_radial_query("RA", "DEC", {image.ra}, {image.dec}, 1.2) AND 
    "PARALLAX" > 0 AND "GAIA_PHOT_G_MEAN_MAG" > 16. 
    '''

    cursor.execute(q)
    result = cursor.fetchall()

    matchra = []
    matchdec = []
    for d in result:
        matchra.append(d[0])
        matchdec.append(d[1])

    matchcoord = SkyCoord(matchra, matchdec, unit='deg')
    catcoord = SkyCoord(catalog.data['X_WORLD'], catalog.data['Y_WORLD'], unit='deg')

    idx, d2d, _ = catcoord.match_to_catalog_sky(matchcoord)
    ind = d2d < 1 * u.arcsec
    catok = catalog.data[ind]

    seeings = []
    for row in catok:
        fwhm = row['FWHM_IMAGE']
        seeings.append(fwhm)

    seeing = np.nanmedian(seeings)
    image.header['SEEING'] = float(seeing)
    image.header_comments['SEEING'] = 'FWHM of seeing in pixels (Goldstein)'
    image.save()
