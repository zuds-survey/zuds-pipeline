
import requests
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from datetime import date
import paramiko
import os
from pathlib import Path
from matplotlib import pyplot as plt

#from baselayer.app.env import load_env
from baselayer.app.model_util import status
from social_tornado.models import TornadoStorage

#from skyportal.models import (init_db, Base, DBSession, ACL, Comment,
#                              Instrument, Group, GroupUser, Photometry, Role,
##                              Source, Spectrum, Telescope, Thumbnail, User,
#                             Token)

#from db import DBSession
import db
from penquins import Kowalski

from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy.visualization import ZScaleInterval

__all__ = ['load_catalog']


CUTOUT_SIZE = 61  # pix
SEARCH_RADIUS = 2. / 3600.  # 2 arcsec
#DB_FTP_DIR = os.getenv('DB_FTP_DIR')
DB_FTP_DIR = '/skyportal/static/thumbnails'
DB_FTP_ENDPOINT = os.getenv('DB_FTP_ENDPOINT')
DB_FTP_USERNAME = 'root'
DB_FTP_PASSWORD = 'root'
DB_FTP_PORT = 222

lookup = dict(zip(range(0, 25), 'abcdefghijklmnopqrstuvwxyz'))

def get_next_name():
    seqquery = "SELECT nextval('namenum')"
    num = db.DBSession().execute(seqquery).fetchone()[0]
    name = 'ZTFC' + str(date.today().year)[2:] + num_to_alpha(num)
    return name

def num_to_alpha(num):
    updates = []
    while num > 0:
        mod = num % 25
        updates.append(lookup[mod])
        num //= 25

    return ''.join(updates[::-1])


def make_stamp(name, ra, dec, vmin, vmax, data, wcs, save=True,
               size=CUTOUT_SIZE):
    coord = SkyCoord(ra, dec, frame='icrs', unit='deg')
    cutout = Cutout2D(data, coord, size, wcs=wcs, fill_value=0.)

    if save:
        plt.imsave(name, np.flipud(cutout.data), vmin=vmin, vmax=vmax,
                   cmap='gray')
        os.chmod(name, 0o774)
    else:
        return cutout


def annotate(source):
    # currsource = source
    # for source in sources[sources.index(currsource):]:

    kusername = os.getenv('KOWALSKI_USERNAME')
    kpassword = os.getenv('KOWALSKI_PASSWORD')
    authtoken = os.getenv('SKYPORTAL_TOKEN')
    headers = {'Authorization': f'token {authtoken}'}
    k = Kowalski(username=kusername, password=kpassword)

    result = k.query({"query_type": "cone_search",
                      "object_coordinates": {
                          "radec": f"[({source.ra}, {source.dec})]",
                          "cone_search_radius": "1.5",
                          "cone_search_unit": "arcsec"
                      }, "catalogs": {
            "Gaia_DR2": {"filter": {}, "projection": {"parallax": 1, "ra": 1, "dec": 1}}
        }})
    rset = result['result_data']['Gaia_DR2'][f'({source.ra}, {source.dec})'.replace('.', '_')]
    if len(rset) > 0:
        parallax = rset[0]['parallax']
        if parallax is not None:
            # this is a variable star
            source.varstar = True
            source.score = -1
            db.DBSession().add(source)
            db.DBSession().commit()

            # write a comment to this effect
            comment = f"Matched to GAIA source id={rset[0]['_id']}"
            r = requests.post('http://***REMOVED***:5000/api/comment', headers=headers,
                              json={'source_id': source.id, 'text': comment, 'attachment': ''})
            if r.status_code != 200:
                raise RuntimeError(r.content)
            else:
                print(r.content)

    result = k.query({"query_type": "cone_search",
                 "object_coordinates": {
                     "radec": f"[({source.ra}, {source.dec})]",
                 "cone_search_radius": "1.5",
                 "cone_search_unit": "arcsec"
             }, "catalogs": {
                 "milliquas_v6": {"filter": {}, "projection": {"ra":1, "dec":1}}
             }})
    rset = result['result_data']['milliquas_v6'][f'({source.ra}, {source.dec})'.replace('.', '_')]
    if len(rset) > 0:

        # write a comment to this effect
        comment = f"Matched to milliquas source id={rset[0]['_id']}"
        r = requests.post('http://***REMOVED***:5000/api/comment', headers=headers,
                          json={'source_id' : source.id, 'text': comment, 'attachment':''})
        if r.status_code != 200:
            raise RuntimeError(r.content)
        else:
            print(r.content)


def load_catalog(catpath, refpath, newpath, subpath):
    """Insert test data"""
    env, cfg = db.load_env()
    basedir = Path(os.path.dirname(__file__))/'..'

    with status(f"Connecting to database {cfg['database']['database']}"):
        db.init_db(**cfg['database'])

    with status("Loading in photometry"):

        with fits.open(catpath) as f, fits.open(newpath) as new:
            data = f[1].data
            imheader = new[0].header

        gooddata = data[data['GOODCUT'] == 1]

        photpoints = []
        triplets = []

        vdict = {}
        zscale = ZScaleInterval()

        pix = []
        wcs = []

        for t, f in zip(('ref','new','sub'), (refpath, newpath, subpath)):
            with fits.open(f) as hdu:
                data = hdu[0].data
                pix.append(data)
                wcs.append(WCS(hdu[0].header))
                vdict[t] = zscale.get_limits(data)

                if t == 'sub':
                    zp = hdu[0].header['MAGZP']


        for row in gooddata:

            ra = row['X_WORLD']
            dec = row['Y_WORLD']
            flux = row['FLUX_AUTO']
            fluxerr = row['FLUXERR_AUTO']

            a_image = float(row['A_IMAGE'])
            b_image = float(row['B_IMAGE'])
            theta_image = float(row['THETA_IMAGE'])

            zpsys = 'ab'



            try:
                obsmjd = imheader['MJDEFF']
            except KeyError:
                obsmjd = imheader['OBSMJD']
            filter = 'ztf' + imheader['FILTERCODE'][-1].lower()
            limmag = imheader['LMT_MG']

            sub_id = os.getenv('subid')

            photpoint = db.StackDetection(ra=ra, dec=dec, flux=flux,
                                          fluxerr=fluxerr, zp=zp, zpsys=zpsys,
                                          filter=filter, mjd=obsmjd, maglimit=limmag,
                                          provenance='gn', method='auto',
                                          subtraction_id=int(sub_id),
                                          a_image=a_image, b_image=b_image,
                                          theta_image=theta_image)


            photpoints.append(photpoint)
            db.DBSession().add(photpoint)
            db.DBSession().commit()

            # do stamps

            stamps = {}

            for key, p, w in zip(['ref', 'new', 'sub'], pix, wcs):
                name = os.path.join(f'/stamps/{photpoint.id}.{key}.png')
                make_stamp(name, photpoint.ra, photpoint.dec, vdict[key][0], vdict[key][1], p, w)
                stamps[f'{key}file'] = name
            triplets.append(stamps)



    with status('Grouping detections into sources'):
        srcquery = 'SELECT ID FROM SOURCES WHERE Q3C_RADIAL_QUERY(RA, DEC, :ra, :dec, 0.0005554)'
        detquery = 'SELECT ID FROM STACKDETECTIONS WHERE Q3C_RADIAL_QUERY(RA, DEC, :ra, :dec, 0.0005554) AND ' \
                   'ID != :id'
        g = db.models.Group.query.first()
        for point in photpoints:
            result = db.DBSession().execute(srcquery, {'ra': point.ra, 'dec': point.dec}).fetchall()
            if len(result) > 0:
                source = db.models.Source.query.get(result[0]['id'])
                point.source = source

                # update source RA and DEC

                points = source.stack_detections
                bestpoint = max(points, key=lambda p: p.flux / p.fluxerr)

                if point.id == bestpoint.id:

                    for t in ['ref', 'new', 'sub']:
                        thumb = db.StackThumbnail(type=t, stackdetection_id=bestpoint.id,
                                                  public_url=f'http://portal.nersc.gov/project/astro250/stamps/{bestpoint.id}.{t}.png')
                        db.DBSession().add(thumb)

                    #source.add_linked_thumbnails()

                    source.ra = bestpoint.ra
                    source.dec = bestpoint.dec

            else:
                result = [a[0] for a in db.DBSession().execute(detquery, {'ra':point.ra, 'dec':point.dec,
                                                                          'id': point.id}).fetchall()]
                if len(result) > 2:
                    # create a new source
                    points = list(db.DBSession().query(db.StackDetection).filter(db.StackDetection.id.in_(result)).all())
                    points = points + [point]

                    bestpoint = max(points, key=lambda p: p.flux / p.fluxerr)

                    ra = bestpoint.ra
                    dec = bestpoint.dec

                    seqquery = "SELECT nextval('namenum')"
                    num = db.DBSession().execute(seqquery).fetchone()[0]
                    name = 'ZTFC' + str(date.today().year)[2:] + num_to_alpha(num)
                    s = db.models.Source(id=name, ra=ra, dec=dec, groups=[g], score=0)

                    for point in points:
                        point.source = s

                    db.DBSession().add(s)
                    db.DBSession().commit()

                    annotate(s)

                    for t in ['ref', 'new', 'sub']:
                        thumb = db.StackThumbnail(type=t, stackdetection_id=bestpoint.id,
                                                  public_url=f'http://portal.nersc.gov/project/astro250/stamps/{bestpoint.id}.{t}.png')
                        db.DBSession().add(thumb)

                    #s.add_linked_thumbnails()

    db.DBSession().commit()


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('catalog')
    parser.add_argument('ref')
    parser.add_argument('new')
    parser.add_argument('sub')
    args = parser.parse_args()

    load_catalog(args.catalog, args.ref, args.new, args.sub)
