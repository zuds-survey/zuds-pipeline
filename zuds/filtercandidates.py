import db
import numpy as np
import os
import sqlalchemy as sa
#import sfdmap

from sqlalchemy import func
from sncosmo.photdata import PhotometricData

import penquins

import sncosmo
from secrets import get_secret

from datetime import datetime, timedelta

C_BOUNDS = [-0.2, 0.2]
X1_BOUNDS = [-1, 1]


def cross_match_source_against_wise_agns(source, k):

    ra, dec = source.ra, source.dec

    # have to do this for geojson
    ra -= 180

    query = """db['AllWISE'].find({'coordinates.radec_geojson': {'$geoWithin': {'$centerSphere' : [[%f, %f], 4.84814e-6]}},
                                   '$expr' : {'$gt': [{'$subtract': ['$w1mpro', '$w2mpro']}, {'$multiply': [0.662,
                                   {'$exp': {'$multiply': [0.232, {'$pow': 
                                   [{'$subtract': ['$w2mpro', 13.97]}, 2]}]}}]}]}})"""
    query = query % (ra, dec)

    while True:
        query_result = k.query({'query_type': 'general_search', 'query': query})
        if query_result['status'] == 'failed':
            print('failed')
            continue
        else:
            break

    matches = query_result['result_data']['query_result']
    return matches


def cross_match_source_against_milliquas(source, k):

    ra, dec = source.ra, source.dec

    # have to do this for geojson
    ra -= 180

    query = """db['milliquas_v6'].find({'coordinates.radec_geojson': 
    {'$geoWithin': {'$centerSphere' : [[%f, %f], 4.84814e-6]}}})"""

    query = query % (ra, dec)

    while True:
        query_result = k.query({'query_type': 'general_search', 'query': query})
        if query_result['status'] == 'failed':
            print('failed')
            continue
        else:
            break

    matches = query_result['result_data']['query_result']
    return matches


def cross_match_source_against_gaia(source, k):

    ra, dec = source.ra, source.dec

    query = {"query_type": "cone_search",
             "object_coordinates": {
                 "radec": f"[({ra}, {dec})]",
                 "cone_search_radius": "1.5",
                 "cone_search_unit": "arcsec"
             },
             "catalogs": {
                 "Gaia_DR2": {"filter": {}, "projection": {"parallax": 1, "ra":1, "dec":1}}
             }}

    while True:
        result = k.query(query)
        if 'status' not in result:
            print(f'failed, {result}, {object.__dict__}',
                  flush=True)
            break

        if result['status'] == 'failed':
            print('failed')
            continue
        else:
            break

    rset = result['result_data']['Gaia_DR2'][f'({ra}, {dec})'.replace('.', '_')]

    if len(rset) == 0:
        return []
    else:
        parallax = rset[0]['parallax']
        if parallax is not None:
            return rset
        else:
            return []


def cross_match_source_against_lensdbs(source):

    ra, dec = source.ra, source.dec
    res = db.DBSession().query(db.DR8North)\
        .filter(db.sa.func.q3c_radial_query(db.DR8North.ra, db.DR8North.dec, ra, dec, 0.0002777 * 10))\
        .order_by(db.sa.func.q3c_dist(db.DR8North.ra, db.DR8North.dec, ra, dec).asc())

    neighb = res.first()

    if neighb is not None:
        g = neighb.gmag
        r = neighb.rmag
        w1 = neighb.w1mag
        z = neighb.z_phot_median
        dz = neighb.z_phot_std
        c1 = (g - r) > 0.84 + 0.44 * (r - w1)
        c2 = (g - r) > 1.5
        c3 = (g - r) > 1.
        c4 = neighb.type.strip() == 'DEV'
        c5 = dz / (1 + z) < 0.5

        if (c1 or c2 or c4) and c3 and c5:
            return neighb

    return None


if __name__ == '__main__':

    db.init_db()

    # get time of previous filter run

    prevtime = db.DBSession().query(func.max(db.FilterRun.tend)).first()[0]
    if prevtime is None:
        prevtime = datetime.now() - timedelta(weeks=9999)

    # get the sources that should have their light curves refit

    v1 = db.sa.func.max(db.models.Photometry.modified).label('modified')
    phot_query = db.DBSession().query(db.models.Photometry.source_id, v1) \
        .group_by(db.models.Photometry.source_id).subquery()

    q3c_join = db.sa.func.q3c_join(db.models.Source.ra, db.models.Source.dec,
                                   db.DR8North.ra, db.DR8North.dec, 0.0002777 * 2)

    ctable = db.DBSession().query(db.models.Source,
                                  db.DR8North).join(
        phot_query, db.models.Source.id == phot_query.c.source_id
    ).filter(
        db.models.Source.modified >= prevtime,
        phot_query.c.modified >= prevtime
    ).join(db.DR8North, q3c_join).options(db.sa.orm.joinedload(db.models.Source.comments))

    final = filter(lambda r: (lambda s, d: d.flux_g > 0 and d.flux_r > 0 and d.flux_w1 > 0 and
                   d.gmag - d.rmag > 1 and (d.z_spec != -99 or (d.z_phot_std / (1 + d.z_phot_median)) < 0.2) and
                                (d.gmag - d.rmag > 0.84 + 0.44 * (d.rmag - d.w1mag) or
                                 d.gmag - d.rmag > 1.5 or
                                 d.type.strip() == 'DEV') and d.type.strip() != 'PSF' and
                              np.isclose(d.parallax, 0.))(r[0], r[1]),
                   ctable)
    # start the run
    run = db.FilterRun(tstart=datetime.now())

    # get dust directory
    #dustmap = sfdmap.SFDMap(os.getenv('SFDMAP_DIR'))

    # connect to kowalski
    while True:
        try:
            k = penquins.Kowalski(username=get_secret('kowalski_username'),
                                  password=get_secret('kowalski_password'))
        except:
            pass
        else:
            break

    for source, galaxy in final:

        light_curve = source.light_curve()

        # remove outliers from the fit
        light_curve = light_curve[light_curve['flux'] / light_curve['fluxerr'] < 100.]

        photdata = PhotometricData(light_curve)

        print(f'matching {source.id}...')

        if len(source.stack_detections) < 3:
            continue

        quasmatches = cross_match_source_against_milliquas(source, k)

        if len(quasmatches) > 0:
            continue

        agnmatches = cross_match_source_against_wise_agns(source, k)

        if len(agnmatches) > 0:
            continue

        # need at least 3 detections > 3 sigma
        if len(light_curve[light_curve['flux'] / light_curve['fluxerr'] >= 3]) < 3:
            continue

        # add in dust
        #dust = sncosmo.CCM89Dust()
        #ebv = dustmap.ebv(source.ra, source.dec)
        #dust['ebv'] = ebv

        # set up the fit model
        fitmod = sncosmo.Model(source='salt2-extended')#, effect_names=['mw'], effect_frames=['obs'])#, effects=[dust])
        fitmod['z'] = galaxy.z_phot_median
        fitmod.set_source_peakabsmag(-18, 'bessellb', 'ab')
        x0_low = fitmod['x0']
        fitmod.set_source_peakabsmag(-20, 'bessellb', 'ab')
        x0_high = fitmod['x0']
        x0_bounds = (x0_low, x0_high)

        # estimate t0
        t0 = db.DBSession().query(db.sa.func.avg(db.StackDetection.mjd)).filter(db.StackDetection.source_id == source.id)
        t0 = t0.first()[0]

        fitmod['t0'] = t0



        # do the fit
        result, fitted_model = sncosmo.fit_lc(photdata, fitmod, vparam_names=['x1', 'c', 'x0', 't0'],
                                              bounds={'x1': X1_BOUNDS, 'c': C_BOUNDS, 'x0': x0_bounds},
                                              guess_t0=False, minsnr=3.)

        for key in result:
            val = result[key]
            if isinstance(val, np.ndarray):
                result[key] = db.psql.array(val.tolist())

        # save it to the DB
        fit = db.Fit(**result)
        fit.source = source
        db.DBSession().add(fit)
        db.DBSession().commit()

        source.score = fit.chisq / fit.ndof
        db.DBSession().add(source)
        db.DBSession().commit()


    """
    except Exception as e:
        run.status = False
        run.reason = e.__str__()
    else:
        run.status = True
    finally:
        run.tend = datetime.now()
        db.DBSession().add(run)
        db.DBSession().commit()
    """

