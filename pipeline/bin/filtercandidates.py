import db
import os
import sqlalchemy as sa
#import sfdmap

from sqlalchemy import func
from sncosmo.photdata import PhotometricData

import penquins

import sncosmo

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


def cross_match_source_against_lensdbs(source, k):

    ra, dec = source.ra, source.dec
    matches = []
    catalogs = ['legacysurveys_photoz_DR6', 'legacysurveys_photoz_DR7', 'sdss_ellipticals']

    # have to do this for geojson
    ra -= 180

    for catalog in catalogs:

        query = "db['%s'].find({'coordinates.radec_geojson' : {'$geoWithin': {'$centerSphere' : [[%f, %f], 4.84814e-6]}}})"
        query = query % (catalog, ra, dec)

        while True:
            query_result = k.query({'query_type': 'general_search', 'query': query})
            if query_result['status'] == 'failed':
                print('failed')
                continue
            else:
                break

        mymatches = query_result['result_data']['query_result']

        if 'legacy' in catalog:
            finalmatches = []
            for match in mymatches:
                g = match['gmag']
                r = match['rmag']
                w1 = match['w1mag']
                z = match['z_phot']
                dz = match['z_phot_err']
                c1 = (g - r) > 0.84 + 0.44 * (r - w1)
                c2 = (g - r) > 1.5
                c3 = (g - r) > 1.
                c4 = match['TYPE'].strip() == 'DEV'
                c5 = dz / (1 + z) < 0.05

                if (c1 or c2 or c4) and c3 and c5:
                    finalmatches.append(match)
            mymatches = finalmatches

        matches.extend(mymatches)

    return matches


if __name__ == '__main__':

    env, cfg = db.load_env()
    db.init_db(**cfg['database'])

    # get time of previous filter run

    prevtime = db.DBSession().query(db.FilterRun.tend.max()).first()
    if prevtime is None:
        prevtime = datetime.now() - timedelta(weeks=9999)

    # get the sources that should have their light curves refit
    sources = db.DBSession().query(db.models.Source).join(db.models.Photometry).filter(
        sa.or_(sa.and_(db.models.Photometry.modified >= prevtime,
                       db.models.Photometry.modified <= func.now()),
               sa.and_(db.models.Source.modified >= prevtime,
                       db.models.Source.modified <= func.now())
               )
    ).distinct(db.models.Source.id).all()  # the distinct part only works in postgres, producing a DISTINCT ON

    # start the run
    run = db.FilterRun(tstart=datetime.now())

    # get dust directory
    dustmap = sfdmap.SFDMap(os.getenv('SFDMAP_DIR'))

    try:
        # connect to kowalski
        while True:
            try:
                k = penquins.Kowalski(username=os.getenv('KOWALSKI_USERNAME'),
                                      password=os.getenv('KOWALSKI_PASSWORD'))
            except:
                pass
            else:
                break

        for source in sources:
            light_curve = source.light_curve()
            photdata = PhotometricData(light_curve)

            # cross match the source against the kowalski catalog
            lensmatches = cross_match_source_against_lensdbs(source, k)

            if len(lensmatches) == 0:
                continue

            quasmatches = cross_match_source_against_milliquas(source, k)

            if len(quasmatches) > 0:
                continue

            agnmatches = cross_match_source_against_wise_agns(source, k)

            if len(agnmatches) > 0:
                continue

            # need at least 2 detections
            if len(light_curve) < 2:
                continue

            # no parallax in gaia
            gaiamatches = cross_match_source_against_gaia(source, k)
            if len(gaiamatches) > 0:
                continue

            # add in dust
            dust = sncosmo.CCM89Dust()
            ebv = dustmap.ebv(source.ra, source.dec)
            dust['ebv'] = ebv

            # set up the fit model
            fitmod = sncosmo.Model(source='salt2-extended', effects=[dust], effect_names=['mw'], effect_frames=['obs'])
            fitmod['z'] = lensmatches[0]['z_phot']
            fitmod.set_source_peakabsmag(-18, 'bessellb', 'ab')
            x0_low = fitmod['x0']
            fitmod.set_source_peakabsmag(-20, 'bessellb', 'ab')
            x0_high = fitmod['x0']
            x0_bounds = (x0_low, x0_high)

            # do the fit
            result, fitted_model = sncosmo.fit_lc(photdata, fitmod, vparam_names=['x1', 'c', 'x0', 't0'],
                                                  bounds={'x1': X1_BOUNDS, 'c': C_BOUNDS, 'x0': x0_bounds})


            # save it to the DB
            fit = db.Fit(**result)
            fit.source = source
            db.DBSession().add(fit)
            db.DBSession().commit()

            source.score = fit.chisq / fit.ndof
            db.DBSession().add(source)
            db.DBSession().commit()


    except Exception as e:
        run.status = False
        run.reason = e.__str__()
    else:
        run.status = True
    finally:
        run.tend = datetime.now()
        db.DBSession().add(run)
        db.DBSession().commit()

