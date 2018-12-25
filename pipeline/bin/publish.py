import requests
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import scipy.misc
from uuid import uuid4
from astropy.time import Time
from datetime import date


__all__ = ['publish_to_marshal']


CUTOUT_SIZE = 201  # pix
SEARCH_RADIUS = 2. / 3600. # 2 arcsec

lookup = dict(zip(range(0, 25), 'abcdefghijklmnopqrstuvwxyz'))

def num_to_alpha(num):
    updates = []
    while num > 0:
        mod = num % 25
        updates.append(lookup[mod])
        num //= 25

    return ''.join(updates)

class Object(object):

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

                xind = np.arange(xp - CUTOUT_SIZE // 2, xp + CUTOUT_SIZE // 2, dtype=int)
                yind = np.arange(yp - CUTOUT_SIZE // 2, yp + CUTOUT_SIZE // 2, dtype=int)

                for m, i in enumerate(xind):
                    if i < 0 or i >= f[0].header['NAXIS1']:
                        continue
                    for n, j in enumerate(yind):
                        if j < 0 or j >= f[0].header['NAXIS2']:
                            continue
                        stamp[n, m] = f[0].section[j, i]

                name = f'/stamps/{objname}.{key}.jpg'
                scipy.misc.imsave(name, stamp)
                stamps[f'{key}file'] = (f'{objname}.{key}.jpg', open(name, 'rb'), 'image/jpeg')

        return stamps


def publish_to_marshal(object, username, password, cursor):

    # insert the object into the database
    objquery = 'INSERT INTO OBJECT (RA, DEC, FILTER, MAG, MAG_ERR, MJD) VALUES ' \
               '(%s, %s, %s, %s, %s, %s) RETURNING ID'
    cursor.execute(objquery, (object.ra, object.dec, object.filter, object.mag, object.magerr,
                              object.mjd))
    objid = cursor.fetchall()[0][0]

    # now see if the object should be associated with a candidate
    assocquery = 'SELECT NAME FROM CANDIDATE WHERE Q3C_RADIAL_QUERY(RA, DEC, %s, %s, %s)'
    cursor.execute(assocquery, (object.ra, object.dec, SEARCH_RADIUS))
    result = cursor.fetchall()

    if len(result) == 0:
        # we need to make a new candidate - get the name of the latest candidate
        query = 'SELECT MAX(NAME) FROM CANDIDATE'
        cursor.execute(query)
        result = cursor.fetchall()[0][0]

        # get the current number
        seqquery = "SELECT nextval('namenum')"
        cursor.execute(seqquery)
        num = cursor.fetchone()[0]
        #name = 'abce'
        name = 'DG' + str(date.today().year)[2:] + num_to_alpha(num)

        # insert the candidate into the db
        query = 'INSERT INTO CANDIDATE (NAME, RA, DEC) VALUES (%s, %s, %s)'
        cursor.execute(query, (name, object.ra, object.dec))
        cursor.connection.commit()

        endpoint = 'http://skipper.caltech.edu:8080/cgi-bin/growth/add_source_atel.cgi'
        stamps = object.make_stamps(name)
        payload = {'name': name, 'ra': object.ra, 'dec': object.dec, 'sciencepids': '60', 'commit':'yes'}
        r = requests.post(endpoint, data=payload, auth=(username, password), files=stamps)

        if r.status_code == 200:
            print('Upload successful')
        else:
            raise ValueError(f'Upload of candidate unsuccessful')

    else:
        name = result[0][0]

    namequery = 'SELECT ID FROM CANDIDATE WHERE NAME=%s'
    cursor.execute(namequery, (name,))
    candid = cursor.fetchone()[0]

    upquery = 'INSERT INTO ASSOC (OBJECT_ID, CANDIDATE_ID) VALUES (%s, %s)'
    cursor.execute(upquery, (objid, candid))

    # now upload the phot - first scrape the marhsal for the sourceid
    r = requests.get('http://skipper.caltech.edu:8080/cgi-bin/growth/view_source.cgi', params={'name': name},
                     auth=(username, password))
    html = r.content
    marshalid = int(html.decode('ascii').split('?sourceid=')[1].split('&')[0])

    photendpoint = f'http://skipper.caltech.edu:8080/cgi-bin/growth/edit_phot.cgi?sourceid={marshalid}'

    jd = Time(object.mjd, format='mjd', scale='utc').jd

    # construct the json to add
    payload = {'commit':'yes',
               'id':"-1",
               'programid':"-1",
               'instrumentid':"2",
               'ra': object.ra,
               'dec': object.dec,
               'era': 0.,
               'edec': 0.,
               'jd': jd,
               'exptime':'-1',
               'filter':object.filter.lower(),
               'magpsf': object.mag,
               'sigmamagpsf': object.magerr,
               'limmag': '-1',
               'issub': 'yes',
               'refsys': 'AB',
               'observer': '-1',
               'reducedby': '-1'}

    r = requests.post(photendpoint, data=payload, auth=(username, password))
    print('photometry upload ' + ('succesful' if r.status_code == 200 else 'unsuccessful'))
