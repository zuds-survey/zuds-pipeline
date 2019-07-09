import db
import os
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.dialects import postgresql as psql
import penquins


ARCSEC = 0.0002777
C_BOUNDS = [-0.2, 0.2]
X1_BOUNDS = [-1, 1]
CHUNKSIZE = 1000

Base = declarative_base()
HitSession = scoped_session(sessionmaker())


def init_db(user, database, password=None, host=None, port=None):
    url = 'postgresql://{}:{}@{}:{}/{}'
    url = url.format(user, password or '', host or '', port or '', database)

    conn = sa.create_engine(url, client_encoding='utf8')

    HitSession.configure(bind=conn)
    Base.metadata.bind = conn

    return conn


class Hit(Base):
    
    __tablename__ = 'hits'

    id = sa.Column(sa.Integer, primary_key=True)
    ra = sa.Column(psql.DOUBLE_PRECISION)
    dec = sa.Column(psql.DOUBLE_PRECISION)
    jdmin = sa.Column(psql.DOUBLE_PRECISION)
    jdmax = sa.Column(psql.DOUBLE_PRECISION)
    cnt = sa.Column(sa.Integer)
    days = sa.Column(sa.Integer)
    q3c = sa.Index("hits_q3c_idx", sa.func.q3c_ang2ipix(ra, dec))


def cross_match_source_against_hits(source):
    radq = sa.func.q3c_radial_query(Hit.ra, Hit.dec, source.ra, source.dec, 3 * ARCSEC)
    matches = HitSession().query(Hit).filter(radq).all()
    return matches


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


if __name__ == '__main__':

    env, cfg = db.load_env()
    db.init_db(**cfg['database'])

    while True:
        try:
            k = penquins.Kowalski(username=os.getenv('KOWALSKI_USERNAME'),
                                  password=os.getenv('KOWALSKI_PASSWORD'))
        except:
            pass
        else:
            break

    # initialize the HITS database at nersc
    init_db(user=os.getenv('HITS_DBUSERNAME'), database=os.getenv('HITS_DBNAME'),
            password=os.getenv('HITS_DBPASSWORD'), host=os.getenv('HITS_DBHOST'),
            port=os.getenv('HITS_DBPORT'))

    while True:

        # get the sources that should have their lens nature checked
        sources = db.DBSession().query(db.PittObject).filter(
            db.PittObject.needs_check == True
        ).with_for_update(skip_locked=True).limit(CHUNKSIZE).all()

        if len(sources) == 0:
            break

        # connect to kowalski

        for source in sources:

            quasmatches = cross_match_source_against_milliquas(source, k)
            source.milliquasmatch = len(quasmatches) > 0

            agnmatches = cross_match_source_against_wise_agns(source, k)
            source.wisematch = len(agnmatches) > 0

            gaiamatches = cross_match_source_against_gaia(source, k)
            source.gaiamatch = len(gaiamatches) > 0

            hitsmatches = cross_match_source_against_hits(source)
            source.hitsmatch = len(hitsmatches) > 0

            db.DBSession().add(source)

        db.DBSession().commit()
