import zuds
zuds.init_db(timeout=60000)
import pandas as pd
import sys
import os
import time
from astropy.io import fits
from astropy.wcs import WCS
from sqlalchemy.dialects.postgresql import array
from functools import wraps
import errno
import signal
import sqlalchemy as sa


zuds.init_db()

__author__ = 'Danny Goldstein <danny@caltech.edu>'
__whatami__ = 'Do the photometry for ZUDS.'

infile = sys.argv[1]  # file listing all the subs to do photometry on
outfile = sys.argv[2] # file listing all the photometry to load into the DB


# get the work
imgs = zuds.get_my_share_of_work(infile)
imgs = sorted(imgs, key=lambda s: s[0].split('ztf_')[1].split('_')[0],
              reverse=True)

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

    jcond2 = sa.and_(
        zuds.ForcedPhotometry.image_id == image_id,
        zuds.ForcedPhotometry.source_id == zuds.Source.id
    )

    query = zuds.DBSession().query(
        zuds.Source.id,
        zuds.Source.ra,
        zuds.Source.dec
    ).outerjoin(
        zuds.ForcedPhotometry, jcond2
    ).filter(
        zuds.ForcedPhotometry.id == None
    ).filter(
        sa.func.q3c_poly_query(
            zuds.Source.ra,
            zuds.Source.dec,
            poly
        )
    )

    return query.all()


@timeout(60)
def get_wcs(fn):
    with fits.open(fn) as hdul:
        hd = hdul[0].header
        wcs = WCS(hd)
    return wcs


safe_raw_ap = timeout(100)(zuds.raw_aperture_photometry)


output = []
start = time.time()

for g, (fn, imgid) in enumerate(imgs):

    now = time.time()

    if now - start > 3600 * 0.75:  # 45 minutes
        break

    maskname = fn.replace('.fits', '.mask.fits')
    rmsname = fn.replace('.fits', '.rms.fits')

    if not (os.path.exists(fn) and os.path.exists(maskname) and os.path.exists(rmsname)):
        print(f'{fn}, {maskname}, and {rmsname} do not all exist, continuing...', flush=True)
        continue

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

    zuds.print_time(nstart, nstop, fn, 'unphotometered sources')


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
                 'dec':dec[k],
                 'zp': row['zp'],
                 'filtercode': row['filtercode'],
                 'obsjd': row['obsjd']}

            output.append(p)

        pstop = time.time()
        zuds.print_time(pstart, pstop, fn, 'actual force photometry')

    except Exception as e:
        print(e)
        continue


if zuds.has_mpi():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # avoid pandas to csv bottleneck using parallelism
    df = pd.DataFrame(output)
    df.to_csv(f'output_{rank:04d}.csv', index=False, header=rank==0)
    comm.Barrier()

    if rank == 0:
        with open(outfile, 'w') as f:
            for fn in [f'output_{r:04d}.csv' for r in range(size)]:
                if os.path.exists(fn):
                    with open(fn, 'r') as g:
                        f.write(g.read())
                os.remove(fn)

        jobid = os.getenv('SLURM_JOB_ID')
        if jobid is not None:
            job = zuds.DBSession().query(zuds.ForcePhotJob).filter(
                zuds.ForcePhotJob.slurm_id == jobid
            ).first()
            job.status = 'ready_for_loading'
            zuds.DBSession().add(job)
            zuds.DBSession().commit()

else:
    df = pd.DataFrame(output)
    df.to_csv(outfile, index=False)

stop = time.time()
zuds.print_time(start, stop, 0, 'start to finish')
