import db
import numpy as np
import pandas as pd
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
from itertools import chain

from io import StringIO


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


def write_csv(output):
    df = pd.DataFrame(output)
    df.to_csv(f'output.csv', index=False)


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


output = []
start = time.time()

for g, (fn, imgid) in enumerate(imgs):

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

        for k, row in enumerate(phot_table):
            p = {'source_id': needed[k][0],
                 'image_id': imgid,
                 'flux': row['flux'],
                 'fluxerr':row['fluxerr'],
                 'flags':int(row['flags']),
                 'ra':ra[k],
                 'dec':dec[k]}

            output.append(p)

        pstop = time.time()
        print_time(pstart, pstop, fn, 'actual force photometry')


    except Exception as e:
        print(e)
        continue


if mpi.has_mpi():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # avoid pandas to csv bottleneck using parallelism
    df = pd.DataFrame(output)
    buf = StringIO()

    df.to_csv(buf, index=False, header=rank == 0)
    csvstr = buf.getvalue()

    output = comm.gather(output, root=0)

    if rank == 0:
        output = '\n'.join(output)
        with open('output.csv', 'w') as f:
            f.write(output)

else:
    df = pd.DataFrame(output)
    df.to_csv('output.csv', index=False)

stop = time.time()
print_time(start, stop, 0, 'start to finish')



