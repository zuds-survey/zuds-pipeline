import db
import numpy as np
import sys
import mpi
import os
import time
import archive
from datetime import datetime, timedelta
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from sqlalchemy.dialects.postgresql import array


from photometry import raw_aperture_photometry

fmap = {1: 'zg',
        2: 'zr',
        3: 'zi'}

db.init_db()
#db.DBSession().autoflush = False
#db.DBSession().get_bind().echo = True

__author__ = 'Danny Goldstein <danny@caltech.edu>'
__whatami__ = 'Do the photometry for ZUDS.'

infile = sys.argv[1]  # file listing all the subs to do photometry on


# get the work
imgs = mpi.get_my_share_of_work(infile)
imgs = sorted(imgs, key=lambda s: s[0].split('ztf_')[1].split('_')[0],
              reverse=True)

def print_time(start, stop, obj, stepname):
    print(f'took {stop-start:.2f} seconds to do {stepname} on {obj}')


def unphotometered_sources(image_id, footprint):

    poly = array(tuple(footprint.ravel()))

    jcond2 = db.sa.and_(
        db.ForcedPhotometry.image_id == image_id,
        db.ForcedPhotometry.source_id == db.models.Source.id
    )

    query = db.DBSession().query(
        db.models.Source.id,
        db.models.Source.ra,
        db.models.Source.dec
    ).outerjoin(
        db.ForcedPhotometry, jcond2
    ).filter(
        db.ForcedPhotometry.id == None
    ).filter(
        db.sa.func.q3c_poly_query(
            db.models.Source.ra,
            db.models.Source.dec,
            poly
        )
    )#.with_for_update(of=models.Source)

    return query.all()


phot = []

for fn, imgid in imgs:

    start = time.time()
    maskname = fn.replace('.fits', '.mask.fits')
    rmsname = fn.replace('.fits', '.rms.fits')

    with fits.open(fn) as hdul:
        hd = hdul[0].header
        wcs = WCS(hd)

    needed = unphotometered_sources(int(imgid), wcs.calc_footprint())

    if len(needed) == 0:
        stop = time.time()
        print(f'phot: no photometry needed on {fn},'
              f' all done (in {stop-start:.2f} sec)')
        continue

    try:
        pstart = time.time()

        ra = [s[1] for s in needed]
        dec = [s[2] for s in needed]
        phot_table = raw_aperture_photometry(fn, rmsname, maskname, ra, dec,
                                             apply_calibration=False)

        myphot = []
        for k, row in enumerate(phot_table):
            p = db.ForcedPhotometry(source_id=needed[k][0],
                                    image_id=imgid,
                                    flux=row['flux'],
                                    fluxerr=row['fluxerr'],
                                    flags=int(row['flags']),
                                    ra=ra[k],
                                    dec=dec[k])
            myphot.append(p)

        pstop = time.time()
        print_time(pstart, pstop, fn, 'actual force photometry')

    except Exception as e:
        print(e)
        continue

    phot.extend(myphot)
    stop = time.time()
    print_time(start, stop, fn, 'start to finish')

#db.DBSession().add_all(phot)

dbstart = time.time()

gtups = [str((p.source_id, p.image_id, 'now()', 'now()', p.flux, p.fluxerr, 'photometry'))
         for p in phot]

if len(gtups) > 0:

    pid = [row[0] for row in db.DBSession().execute(
        'INSERT INTO objectswithflux (source_id, image_id, created_at, modified, '
        'flux, fluxerr, type) '
        f'VALUES {",".join(gtups)} RETURNING ID'
    )]

    ftups = [str((i, p.flags, p.ra, p.dec))  for i, p in zip(pid, phot)]
    db.DBSession().execute(f'INSERT INTO forcedphotometry (id, flags, ra, dec) '
                           f'VALUES {",".join(ftups)}')

    db.DBSession().commit()
    dbstop = time.time()

    print(f'phot: took {dbstop-dbstart:.2f} sec to do db insert', flush=True)
else:
    print('nothing to push to the database.')








