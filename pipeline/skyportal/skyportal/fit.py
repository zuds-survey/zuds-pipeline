import numpy as np
import sncosmo
from sncosmo.photdata import PhotometricData
from astropy.table import Table
from sqlalchemy.sql.expression import func
from .models import DBSession, Source, SDSSObject, CFHTObject, RongpuObject

GALAXY_TYPES = [CFHTObject, SDSSObject, RongpuObject]
TWOARCSEC_DEG = 0.0002777 * 2.


def get_source(source_or_sourceid):

    if isinstance(source_or_sourceid, Source):
        source = source_or_sourceid
    else:
        try:
            source = DBSession().query(Source).filter(Source.id == source_or_sourceid).first()
        except KeyError:
            raise TypeError(f'"{source_or_sourceid}" not a valid source or sourceid.')
    return source


def get_photometry(source_or_sourceid):

    source = get_source(source_or_sourceid)

    points = []
    for point in source.forcedphotometry:
        points.append({
            'time': point.mjd,
            'filter': point.filter,
            'flux': point.flux,
            'fluxerr': point.fluxerr,
            'zp': point.zeropoint,
            'zpsys': 'ab'
        })
    table = Table(points)
    return PhotometricData(table)


def get_nearest_neighbor(object, table_class):

    neighbor = DBSession()\
        .query(table_class)\
        .filter(func.q3c_radial_query(table_class.ra, table_class.dec, object.ra, object.dec, TWOARCSEC_DEG))\
        .order_by(func.q3c_dist(table_class.ra, table_class.dec, object.ra, object.dec).asc())\
        .first()

    if neighbor is None:
        raise KeyError(f'Object "{object}" has no neighbor of type "{table_class}" within the search radius.')

    return neighbor


def get_overlapping_galaxy(source_or_sourceid):

    source = get_source(source_or_sourceid)

    for gtype in GALAXY_TYPES:
        try:
            galaxy = get_nearest_neighbor(source, gtype)
        except KeyError:
            continue
        else:
            return galaxy

    raise KeyError(f'No neighbors found for source "{source}."')


def fit_source(source_or_sourceid, fix_z_to_nearest_neighbor=True, bounds=None):

    source = get_source(source_or_sourceid)
    photom = get_photometry(source)

    vparam_names = ['x0', 'x1', 'c', 'z']
    fitmod = sncosmo.Model(source='salt2')

    if fix_z_to_nearest_neighbor:
        neighbor = get_overlapping_galaxy(source)
        zfix = neighbor.redshift
        vparam_names.pop(-1)
        fitmod['z'] = zfix

    # this is where the magic happens
    result = sncosmo.fit_lc(data=photom, model=fitmod, vparam_names=vparam_names, bounds=bounds)
    return result
