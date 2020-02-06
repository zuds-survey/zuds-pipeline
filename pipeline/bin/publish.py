
import requests
import numpy as np
from datetime import date
import os
from matplotlib import pyplot as plt
import db
from penquins import Kowalski

from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy.visualization import ZScaleInterval

__all__ = ['load_catalog']


CUTOUT_SIZE = 63  # pix
SEARCH_RADIUS = 2. / 3600.  # 2 arcsec
#DB_FTP_DIR = os.getenv('DB_FTP_DIR')
DB_FTP_DIR = '/skyportal/static/thumbnails'
DB_FTP_ENDPOINT = os.getenv('DB_FTP_ENDPOINT')
DB_FTP_USERNAME = 'root'
DB_FTP_PASSWORD = 'root'
DB_FTP_PORT = 222

lookup = dict(zip(range(0, 25), 'abcdefghijklmnopqrstuvwxyz'))

def get_next_name(num=None):
    if num is None:
        seqquery = "SELECT nextval('namenum')"
        num = db.DBSession().execute(seqquery).fetchone()[0]
    name = 'ZUDS' + str(date.today().year)[2:] + num_to_alpha(num)
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


