import requests
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import scipy.misc
from astropy.time import Time
from datetime import date
from ftplib import FTP
import os
from pathlib import Path

from baselayer.app.env import load_env
from baselayer.app.model_util import status

from skyportal.models import (init_db, Base, DBSession, ACL, Comment,
                              Instrument, Group, GroupUser, Photometry, Role,
                              Source, Spectrum, Telescope, Thumbnail, User,
                              Token)

__all__ = ['publish_to_marshal']


CUTOUT_SIZE = 51  # pix
SEARCH_RADIUS = 2. / 3600.  # 2 arcsec
DB_FTP_DIR = os.getenv('DB_FTP_DIR')
DB_FTP_ENDPOINT = os.getenv('DB_FTP_ENDPOINT')
DB_FTP_USERNAME = os.getenv('DB_FTP_USERNAME')
DB_FTP_PASSWORD = os.getenv('DB_FTP_PASSWORD')

lookup = dict(zip(range(0, 25), 'abcdefghijklmnopqrstuvwxyz'))


def num_to_alpha(num):
    updates = []
    while num > 0:
        mod = num % 25
        updates.append(lookup[mod])
        num //= 25

    return ''.join(updates)


class Detection(object):

    def __init__(self, ra, dec, mag, magerr, mjd, filter, locdict):

        self.ra = ra
        self.dec = dec
        self.locdict = locdict
        self.mjd = mjd
        self.mag = mag
        self.magerr = magerr
        self.filter = filter

    def make_stamps(self, objname):

        stamps = {}

        for key in self.locdict:
            fname = self.locdict[key]

            with fits.open(fname) as f:

                wcs = WCS(f[0].header)
                xp, yp = wcs.all_world2pix([[self.ra, self.dec]], 0)[0]

                stamp = np.zeros((CUTOUT_SIZE, CUTOUT_SIZE))

                xind = np.arange(xp - CUTOUT_SIZE // 2, xp + CUTOUT_SIZE // 2 + 1, dtype=int)
                yind = np.arange(yp - CUTOUT_SIZE // 2, yp + CUTOUT_SIZE // 2 + 1, dtype=int)

                for m, i in enumerate(xind):
                    if i < 0 or i >= f[0].header['NAXIS1']:
                        continue
                    for n, j in enumerate(yind):
                        if j < 0 or j >= f[0].header['NAXIS2']:
                            continue
                        stamp[n, m] = f[0].section[j, i]

                name = f'/stamps/{objname}.{key}.png'
                scipy.misc.imsave(name, stamp)
                stamps[f'{key}file'] = (f'{objname}.{key}.png', open(name, 'rb'), 'image/png')

        return stamps


def load_catalog(catpath, refpath, newpath, subpath):
    """Insert test data"""
    env, cfg = load_env()
    basedir = Path(os.path.dirname(__file__))/'..'

    with status(f"Connecting to database {cfg['database']['database']}"):
        init_db(**cfg['database'])

    with status("Loading in photometry"):

        with fits.open(catpath) as f:
            data = f[1].data
            imheader = f[2].header

        gooddata = data[data['GOODCUT'] == 1]

        p48 = DBSession().query(Instrument).filter('p48' in Instrument.name).first()
        photpoints = []
        triplets = []

        for row in gooddata:

            ra = row['X_WORLD']
            dec = row['Y_WORLD']
            mag = row['MAG_BEST']
            e_mag = row['MAGERR_BEST']
            obsmjd = imheader['MJD-OBS']
            obsjd = Time(obsmjd, format='mjd', scale='utc').jd
            filter = imheader['FILTER']

            photpoint = Photometry(instrument=p48, ra=ra, dec=dec, mag=mag,
                                   e_mag=e_mag, jd=obsjd, filter=filter)
            photpoints.append(photpoint)

            # do stamps
            detection = Detection(ra, dec, mag, e_mag, obsmjd, filter,
                                  {'ref':refpath, 'new': newpath, 'sub':subpath})

            stamps = detection.make_stamps(photpoint.id)
            triplets.append(stamps)

        DBSession().add_all(photpoints)

    with status("Uploading stamps"):
        with FTP(DB_FTP_ENDPOINT, user=DB_FTP_USERNAME, passwd=DB_FTP_PASSWORD) as ftp:
            ftp.cwd(DB_FTP_DIR)
            for triplet, photpoint in zip(triplets, photpoints):
                for key in triplet:
                    f = triplet[key]
                    ftp.put(f)
                    remotename = os.path.join(DB_FTP_DIR, os.path.basename(f))
                    thumb = Thumbnail(type=key[:3], photometry_id=photpoint.id,
                                      file_uri=remotename, public_url=os.path.join('/', remotename))
                    DBSession().add(thumb)


    with status('Grouping detections into sources'):
        srcquery = 'SELECT ID FROM SOURCES WHERE Q3C_RADIAL_QUERY(RA, DEC, :ra, :dec, 2.)'
        detquery = 'SELECT ID FROM PHOTOMETRY WHERE Q3C_RADIAL_QUERY(RA, DEC, :ra, :dec, 2.) AND ' \
                   'ID != :id'
        for point in photpoints:
            result = DBSession().execute(srcquery, {'ra': point.ra, 'dec': point.dec}).fetchall()
            if len(result) > 0:
                source = Source.query.get(result[0]['id'])
                point.source = source

                # update source RA and DEC
                source.ra = np.median([p.ra for p in source.photometry])
                source.dec = np.median([p.dec for p in source.photometry])


            else:
                result = DBSession().execute(detquery, {'ra':point.ra, 'dec':point.dec,
                                                        'id': point.id}).fetchall()
                if len(result) > 0:
                    # create a new source
                    points = list(Photometry.query.get(result['id'])) + [point]
                    ra = np.median([p.ra for p in points])
                    dec = np.median([p.dec for p in points])

                    seqquery = "SELECT nextval('namenum')"
                    num = DBSession().execute(seqquery).fetchone()[0]
                    name = 'ZTFC' + str(date.today().year)[2:] + num_to_alpha(num)
                    s = Source(id=name, ra=ra, dec=dec)
                    DBSession.add(s)
                    s.add_linked_thumbnails()

    DBSession().commit()
