
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


def num_to_alpha(num):
    updates = []
    while num > 0:
        mod = num % 25
        updates.append(lookup[mod])
        num //= 25

    return ''.join(updates[::-1])


def make_stamp(name, ra, dec, vmin, vmax, data, wcs):
    coord = SkyCoord(ra, dec, frame='icrs', unit='deg')
    cutout = Cutout2D(data, coord, CUTOUT_SIZE, wcs=wcs, fill_value=0.)
    plt.imsave(name, np.flipud(cutout.data), vmin=vmin, vmax=vmax, cmap='gray')
    os.chmod(name, 0o774)


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
                                          subtraction_id=int(sub_id))

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



                source.ra = np.median([p.ra for p in source.stack_detections])
                source.dec = np.median([p.dec for p in source.stack_detections])


            else:
                result = [a[0] for a in db.DBSession().execute(detquery, {'ra':point.ra, 'dec':point.dec,
                                                                          'id': point.id}).fetchall()]
                if len(result) > 2:
                    # create a new source
                    points = list(db.DBSession().query(db.StackDetection).filter(db.StackDetection.id.in_(result)).all())
                    points = points + [point]
                    ra = np.median([p.ra for p in points])
                    dec = np.median([p.dec for p in points])

                    seqquery = "SELECT nextval('namenum')"
                    num = db.DBSession().execute(seqquery).fetchone()[0]
                    name = 'ZTFC' + str(date.today().year)[2:] + num_to_alpha(num)
                    s = db.models.Source(id=name, ra=ra, dec=dec, groups=[g])

                    for point in points:
                        point.source = s

                    db.DBSession().add(s)
                    db.DBSession().commit()

                    bestpoint = sorted(points, key=lambda p: p.flux / p.fluxerr)[-1]

                    for t in ['ref', 'new', 'sub']:
                        thumb = db.StackThumbnail(type=t, stackdetection_id=bestpoint.id,
                                                  public_url=f'http://portal.nersc.gov/project/astro250/stamps/{firstpoint.id}.{t}.png')
                        db.DBSession().add(thumb)

                    s.add_linked_thumbnails()

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
