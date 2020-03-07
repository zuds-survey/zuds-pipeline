import db
db.DBSession.remove()
db.init_db(timeout=60000)
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
from functools import wraps
import errno
import signal

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

    
class TimeoutError(Exception):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


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


@timeout(60)
def get_wcs(fn):
    with fits.open(fn) as hdul:
        hd = hdul[0].header
        wcs = WCS(hd)
    return wcs

safe_raw_ap = timeout(100)(raw_aperture_photometry)

output = []
start = time.time()

for g, (fn, imgid) in enumerate(imgs):

    now = time.time()

    if now - start > 3600 * 0.5:  # 45 minutes
        break

    maskname = fn.replace('.fits', '.mask.fits')
    rmsname = fn.replace('.fits', '.rms.fits')

    try:
        wcs = get_wcs(fn)
    except TimeoutError:
        print(f'timed out getting wcs on {fn}, continuing...')
        continue

    nstart = time.time()
    try:
        needed = unphotometered_sources(int(imgid), wcs.calc_footprint())
    except Exception as e:
        print(e)
        continue
    nstop = time.time()

    print_time(nstart, nstop, fn, 'unphotometered sources')


    if len(needed) == 0:
        print(f'phot: no photometry needed on {fn},'
              f' all done (in {nstop-nstart:.2f} sec)')
        continue


    try:
        pstart = time.time()

        ra = [s[1] for s in needed]
        dec = [s[2] for s in needed]
        phot_table = safe_raw_ap(fn, rmsname, maskname,
                                 ra, dec, apply_calibration=False)
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
    size = comm.Get_size()

    # avoid pandas to csv bottleneck using parallelism
    df = pd.DataFrame(output)
    df.to_csv(f'output_{rank:04d}.csv', index=False, header=rank==0)
    comm.Barrier()

    if rank == 0:
        with open('output.csv', 'w') as f:
            for fn in [f'output_{r:04d}.csv' for r in range(size)]:
                if os.path.exists(fn):
                    with open(fn, 'r') as g:
                        f.write(g.read())

else:
    df = pd.DataFrame(output)
    df.to_csv('output.csv', index=False)

stop = time.time()
print_time(start, stop, 0, 'start to finish')



