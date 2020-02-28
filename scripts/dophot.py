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
imgs = sorted(imgs, key=lambda s: s.split('ztf_')[1].split('_')[0], reverse=True)

def print_time(start, stop, obj, stepname):
    print(f'took {stop-start:.2f} seconds to do {stepname} on {obj}')


def get_source_data():
    source_data = db.DBSession().query(db.models.Source.id,
                                       db.models.Source.ra,
                                       db.models.Source.dec).all()


    return source_data


if mpi.has_mpi():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        source_data = get_source_data()

    else:
        source_data = None
    source_data = comm.bcast(source_data, root=0)

else:
    source_data = get_source_data()


ids = [s[0] for s in source_data]

source_coords = SkyCoord([s[1] for s in source_data],
                         [s[2] for s in source_data],
                         unit='deg')


phot = []

for fn, imgid in imgs:

    start = time.time()
    maskname = fn.replace('.fits', '.mask.fits')
    rmsname = fn.replace('.fits', '.rms.fits')

    # get the source ids that have already been done
    done = db.DBSession().query(db.ForcedPhotometry.source_id).filter(
        db.ForcedPhotometry.image_id == imgid
    )
    done = [d[0] for d in done]

    with fits.open(fn) as hdul:
        wcs = WCS(hdul[0].header)

    ok = wcs.footprint_contains(source_coords)
    ids_contained = ids[ok]
    needed = np.setdiff1d(ids_contained, done)
    needed_coords = source_coords[ok]

    if len(needed) == 0:
        stop = time.time()
        print(f'phot: no photometry needed on {fn},'
              f' all done (in {stop-start:.2f} sec)')
        continue

    try:
        pstart = time.time()

        ra = [s.ra.deg for s in needed_coords]
        dec = [s.dec.deg for s in needed_coords]
        phot_table = raw_aperture_photometry(fn, rmsname, maskname, ra, dec,
                                             apply_calibration=False)

        myphot = []
        for k, (i, row) in enumerate(phot_table.iterrows()):
            p = db.ForcedPhotometry(source_id=needed[k],
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

gtups = ['(' + str((p.source_id, p.image_id, 'now()', 'now()', p.flux, p.fluxerr, 'photometry'))  + ')'
         for p in phot]

pid = [row[0] for row in db.DBSession().execute(
    'INSERT INTO objectswithflux (source_id, image_id, created_at, modified '
    'flux, fluxerr, type) '
    f'VALUES {",".join(gtups)} RETURNING ID'
)]

ftups = ['(' + str((i, p.flags, p.ra, p.dec))  + ')' for i, p in zip(pid, phot)]
db.DBSession().execute(f'INSERT INTO forcedphotometry (id, flags, ra, dec) '
                       f'VALUES {",".join(ftups)}')

db.DBSession().commit()
dbstop = time.time()

print(f'phot: took {dbstop-dbstart:.2f} sec to do db insert', flush=True)






