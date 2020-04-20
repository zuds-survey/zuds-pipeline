import sys
import numpy as np
import traceback
import math
from astropy.io import fits
import psycopg2
import os
from astropy.coordinates import SkyCoord
from penquins import Kowalski

from .secrets import get_secret

__all__ = ['xmatch']


def logon():
    """ Log onto Kowalski """
    username = get_secret('kowalski_username')
    password = get_secret('kowalski_password')
    for i in range(3):
        try:
            s = Kowalski(
                protocol='https', host='kowalski.caltech.edu', port=443,
                verbose=False, username=username, password=password)
        except Exception as e:
            traceback.print_exception(*sys.exc_info())
            print('continuing..')
            continue
        else:
            return s
    raise e


def private_logon():
    """ Log on to the ztfimages database on private """
    conn = psycopg2.connect(
            host=get_secret('hpss_dbhost'), user=get_secret('hpss_dbusername'),
            password=get_secret('hpss_dbpassword'),
            database=get_secret('olddb'),
            port=get_secret('hpss_dbport'))
    cur = conn.cursor()
    return cur


def abmag(flux):
    """ takes flux as nanomaggies, gives AB mag """
    # don't take the log of 0!
    if flux <= 0:
        return 0
    return -2.5*np.log10(flux)+22.5


def getsgtable(dec):
    """
    Format:
    hlsp_ps1-psc_ps1_gpc1_<declination>_multi_v1_cat.fits

    where <declination> is a string that indicates the lower or upper limit
    of the 1-degree declination strip of sources contained in this file,
    taking on values from "-31" to "89".
    For positive declinations, this integer specifies the lower limit,
    such that file <7> contains sources in the range [7,8].
    For negative declinations, this integer specifies the upper limit,
    such that the file <-7> contains sources in the range [-8,-7].

    Parameters
    ----------
    dec: source dec in decimal degrees

    Returns
    -------
    path and name of file with sgscore table
    """
    # Danny: add path to table here



    if dec >= 0:
        base = f"hlsp_ps1-psc_ps1_gpc1_{math.floor(dec):d}_multi_v1_cat.fits"
    else:
        base = f"hlsp_ps1-psc_ps1_gpc1_{math.floor(dec)+1:d}_multi_v1_cat.fits"

    return os.path.join(get_secret('ps1_dir'), base)

def ps1(s, ra, dec):
    """ Cross-match a position against PS1 DR1

    Warning -- this assumes that there are at least 3 PS1 matches
    within 30 arcsec

    For each source, find the corresponding objID in the sgscore table
    and pull out the sgscore

    Parameters
    ----------
    s: Kowalski logon instance
    ra: RA of the source in decimal degrees
    dec: Dec of the source in decimal degrees

    Returns
    -------
    out: dictionary containing info on closest three matches within 30 arcsec
    """
    # Coordinates of source
    c1 = SkyCoord(ra, dec, unit='deg')

    # Query
    q = {"query_type": "cone_search",
         "object_coordinates": {
             "radec": '[(%s, %s)]' %(ra,dec),
             "cone_search_radius": "30",
             "cone_search_unit": "arcsec"
         },
         "catalogs": {
             "PS1_DR1": {
                 "filter": {},
                 "projection": {
                     "raMean": 1,
                     "decMean": 1,
                     "gMeanPSFMag": 1,
                     "rMeanPSFMag": 1,
                     "iMeanPSFMag": 1,
                     "zMeanPSFMag": 1,
                     }
             }
         },
         "kwargs": {}
         }
    r = s.query(query=q)
    data = r['result_data']['PS1_DR1']
    nmatches = 0
    for key in data:
        matches = np.array(data[key]) # should be only one key
        nmatches = len(matches)

    # Sort by distance
    dist = np.zeros(nmatches)
    for ii,match in enumerate(matches):
       c2 = SkyCoord(match['raMean'], match['decMean'], unit='deg')
       dist[ii] = c1.separation(c2).arcsec
    order = np.argsort(dist)
    closest_matches = matches[order][0:3]

    # Load the sgscore table that corresponds to this declination
    tab = getsgtable(dec)
    with fits.open(tab) as F:
        objid = F[1].data['objid']
        ps_score = F[1].data['ps_score']

    # Return the closest three matches as a dictionary
    out = {}
    for ii,match in enumerate(closest_matches):
        oid = matches[ii]['_id']
        try:
            out['objectidps%s' % (ii + 1)] = oid
        except:
            out['objectidps%s' % (ii + 1)] = -999
        try:
            out['sgscore%s' % (ii + 1)] = float(ps_score[objid==int(oid)][0])
        except:
            out['sgscore%s' % (ii + 1)] = -999
        try:
            out['distpsnr%s' % (ii + 1)] = float(dist[order][ii])
        except:
            out['distpsnr%s' % (ii + 1)] = -999
        try:
            out['psgmag%s' %(ii+1)] = matches[ii]['gMeanPSFMag']
        except:
            out['psgmag%s' %(ii+1)] = -999
        try:
            out['psrmag%s' %(ii+1)] = matches[ii]['rMeanPSFMag']
        except:
            out['psrmag%s' %(ii+1)] = -999
        try:
            out['psimag%s' %(ii+1)] = matches[ii]['iMeanPSFMag']
        except:
            out['psimag%s' %(ii+1)] = -999
        try:
            out['pszmag%s' %(ii+1)] = matches[ii]['zMeanPSFMag']
        except:
            out['pszmag%s' %(ii+1)] = -999

    return out


def legacysurvey(cur, ra, dec, source_id):
    """ Cross-match candidate position with LegacySurvey DR8

    Warning: it assumes that there are at least 3 LegacySurvey matches
    within 30 arcseconds

    Parameters
    ---------
    cur: cursor for navigating DB on private
    ra: RA in decimal degrees
    dec: Dec in decimal degrees

    Return
    ------
    out: dictionary with info on closest 3 sources within 30 arcsec
    """
    table = 'dr8_north_join_neighbors'
    if dec < 32:
        table = 'dr8_south_join_neighbors'
    cur.execute('SELECT sep,'
                '"OBJID","TYPE","RA","DEC","EBV","FLUX_G","FLUX_R", '
                '"FLUX_Z","FLUX_W1","FLUX_W2","FLUX_W3","FLUX_W4", '
                '"GAIA_PHOT_G_MEAN_MAG","PARALLAX", '
                'z_phot_mean,z_phot_median,z_phot_std,z_phot_l68,z_phot_u68,'
                'z_phot_l95,z_phot_u95,z_spec '
                f'FROM {table} where sid={source_id} order by rank '
                f'desc limit 3')
    matches = cur.fetchall()

    out = {}
    for ii,match in enumerate(matches):
        out['lsdistnr%s' %(ii+1)] = matches[ii][0]
        out['lsobjectid%s' %(ii+1)] = matches[ii][1]
        out['lstype%s' %(ii+1)] = matches[ii][2]
        out['lsebv%s' %(ii+1)] = matches[ii][5]
        out['lsg%s' %(ii+1)] = abmag(matches[ii][6])
        out['lsr%s' %(ii+1)] = abmag(matches[ii][7])
        out['lsz%s' %(ii+1)] = abmag(matches[ii][8])
        out['lsw1_%s' %(ii+1)] = abmag(matches[ii][9])
        out['lsw2_%s' %(ii+1)] = abmag(matches[ii][10])
        out['lsw3_%s' %(ii+1)] = abmag(matches[ii][11])
        out['lsw4_%s' %(ii+1)] = abmag(matches[ii][12])
        out['lsgaiag%s' %(ii+1)] = matches[ii][13]
        out['lsgaiap%s' %(ii+1)] = matches[ii][14]
        out['lszphotmean%s' %(ii+1)] = matches[ii][15]
        out['lszphotmed%s' %(ii+1)] = matches[ii][16]
        out['lszphotstd%s' %(ii+1)] = matches[ii][17]
        out['lszphotl68%s' %(ii+1)] = matches[ii][18]
        out['lszphotu68%s' %(ii+1)] = matches[ii][19]
        out['lszphotl95%s' %(ii+1)] = matches[ii][20]
        out['lszphotu95%s' %(ii+1)] = matches[ii][21]
        out['lszspec%s' %(ii+1)] = matches[ii][22]
    return out


def ztfalerts(s, ra, dec):
    """ Cross-match candidate position with ZTF alerts table

    Parameters
    ----------
    s: Kowalski logon instance
    ra: RA in decimal degrees
    dec: Dec in decimal degrees

    Returns
    -------
    ztfname (np array): list of ZTF names for any sources within 1.5 arcseconds
    """
    # Coordinates of source
    c1 = SkyCoord(ra, dec, unit='deg')

    # Query
    q = {"query_type": "cone_search",
         "object_coordinates": {
             "radec": '[(%s, %s)]' %(ra,dec),
             "cone_search_radius": "1.5",
             "cone_search_unit": "arcsec"
         },
         "catalogs": {
             "ZTF_alerts": {
                 "filter": {},
                 "projection": {
                     "objectId": 1,
                     "_id": 0
                     }
             }
         },
         "kwargs": {}
         }
    r = s.query(query=q)
    data = r['result_data']['ZTF_alerts']
    nmatches = 0
    names = []
    for key in data:
        matches = np.array(data[key]) # should be only one key
        for match in matches:
            names.append(match['objectId'])
    names = np.unique(np.array(names))
    return names


def milliquas(s, ra, dec):
    """ Cross-match candidate position with milliquas table

    Parameters
    ----------
    s: Kowalski logon instance
    ra: RA in decimal degrees
    dec: Dec in decimal degrees

    Returns
    -------
    mqname (np array): list of IDs for any milliquas sources within 1.5 arcsec
    """
    # Coordinates of source
    c1 = SkyCoord(ra, dec, unit='deg')

    # Query
    q = {"query_type": "cone_search",
         "object_coordinates": {
             "radec": '[(%s, %s)]' %(ra,dec),
             "cone_search_radius": "1.5",
             "cone_search_unit": "arcsec"
         },
         "catalogs": {
             "milliquas_v6": {
                 "filter": {},
                 "projection": {
                     "Name": 1,
                     "_id": 0
                     }
             }
         },
         "kwargs": {}
         }
    r = s.query(query=q)
    data = r['result_data']['milliquas_v6']
    nmatches = 0
    names = []
    for key in data:
        matches = np.array(data[key]) # should be only one key
        for match in matches:
            names.append(match['Name'])
    names = np.unique(np.array(names))
    return names


def tns(s, ra, dec):
    """ Cross-match candidate position with TNS

    Parameters
    ----------
    s: Kowalski logon instance
    ra: RA in decimal degrees
    dec: Dec in decimal degrees

    Returns
    -------
    mqname (np array): list of IDs for any TNS sources within 1.5 arcsec
    """
    # Coordinates of source
    c1 = SkyCoord(ra, dec, unit='deg')

    # Query
    q = {"query_type": "cone_search",
         "object_coordinates": {
             "radec": '[(%s, %s)]' %(ra,dec),
             "cone_search_radius": "1.5",
             "cone_search_unit": "arcsec"
         },
         "catalogs": {
             "TNS": {
                 "filter": {},
                 "projection": {
                     "name": 1,
                     "_id": 0
                     }
             }
         },
         "kwargs": {}
         }
    r = s.query(query=q)
    data = r['result_data']['TNS']
    nmatches = 0
    names = []
    for key in data:
        matches = np.array(data[key]) # should be only one key
        for match in matches:
            names.append(match['name'])
    names = np.unique(np.array(names))
    return names


def xmatch(ra, dec, source_id):
    """ Cross-match against all necessary catalogs

    Parameters
    ----------
    ra: RA of source in decimal degrees
    dec: Dec of source in decimal degrees

    Returns
    -------
    out: dictionary of alert catalog fields
    """

    # Connect to Kowalski and to private
    s = logon()
    cur = private_logon()

    out = ps1(s, ra, dec)
    out.update(legacysurvey(cur, ra, dec, source_id))
    out['ztfname'] = ','.join(ztfalerts(s, ra, dec))
    out['mqid'] = ','.join(milliquas(s, ra, dec))
    out['tnsid'] = ','.join(tns(s, ra, dec))

    s.close()
    cur.connection.close()

    return out


if __name__=="__main__":
    print(xmatch(100,30))
