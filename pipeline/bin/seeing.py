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

    if catalog is None or not catalog.ismapped:
        catalog = db.PipelineFITSCatalog.from_image(image)

    data = catalog.data
    seeings = []
    for row in data:
        fwhm = row['FWHM_IMAGE']
        seeings.append(fwhm)

    seeing = np.nanmedian(seeings)
    image.header['SEEING'] = float(seeing)
    image.header_comments['SEEING'] = 'FWHM of seeing in pixels (Goldstein)'
    image.save()
