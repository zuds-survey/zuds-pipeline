import db
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from secrets import get_secret
import penquins


def estimate_seeing(image):
    """Estimate the seeing on an image by comparing its catalog to GAIA stars."""

    # connect to kowalski to query gaia
    username = get_secret('kowalski_username')
    password = get_secret('kowalski_password')
    kowalski = penquins.Kowalski(username=username, password=password)
    catalog = image.catalog

    if catalog is None or not catalog.ismapped:
        catalog = db.PipelineFITSCatalog.from_image(image)


    q = {"query_type": "cone_search",
         "object_coordinates": {
             "radec": [(image.ra, image.dec)],
             "cone_search_radius": "1.2",
             "cone_search_unit": "deg"
         },
         "catalogs": {
             "Gaia_DR2": {
                 "filter": {
                     "parallax": {"$gt": 0.},
                     "phot_g_mean_mag": {"$gt": 16.}  # only return stars
                 },
                 "projection": {
                     "_id": 1,
                     "ra": 1,
                     "dec": 1
                 }
             }
         },
         "kwargs": {}
    }
    result = kowalski.query(q)
    if result['status'] != 'done':
        raise ValueError(f'Kowalski Error: {result}')

    stars = result['result_data']['Gaia_DR2']
    matchra = []
    matchdec = []
    for d in stars.values():
        for dd in d:
            matchra.append(dd['ra'])
            matchdec.append(dd['dec'])

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
