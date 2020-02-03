import db
import os
import sys
import mpi
import time
import archive
import pandas as pd
from pathlib import Path


fmap = {1: 'zg',
        2: 'zr',
        3: 'zi'}

db.init_db()
#db.DBSession().get_bind().echo = True

__author__ = 'Danny Goldstein <danny@caltech.edu>'
__whatami__ = 'Make the references for ZUDS.'

infile = sys.argv[1]  # file listing all the directories to build refs for
min_date = pd.to_datetime(sys.argv[2])  # minimum allowable date for refimgs
max_date = pd.to_datetime(sys.argv[3])  # maximum allowable date for refimgs
version = sys.argv[4]

# get the work
my_dirs = mpi.get_my_share_of_work(infile)

# make a reference for each directory
for d in my_dirs:

    t_start = time.time()

    # get all the science images
    image_objects = []
    sci_fns = Path(d).glob('ztf*sciimg.fits')

    # load the objects partway into memory
    ok = []
    for fn in sci_fns:
        sci = db.ScienceImage.get_by_basename(fn.name)
        sci.map_to_local_file(f'{fn}')
        maskname = f'{fn.parent / sci.mask_image.basename}'
        sci.mask_image.map_to_local_file(maskname)

        try:
            sci.load()
        except Exception as e:
            print(f'bad: File {sci.basename} is corrupted, skipping...',
                  flush=True)
            continue
        sci.clear()
        sci.map_to_local_file(f'{fn}')

        try:
            sci.mask_image.load()
        except Exception as e:
            print(f'bad: File {sci.mask_image.basename} is corrupted, '
                  f'skipping...',
                  flush=True)
            continue
        sci.mask_image.clear()
        sci.mask_image.map_to_local_file(maskname)

        c1 = min_date <= sci.obsdate <= max_date
        c2 = 1.7 < sci.seeing < 2.5
        c3 = 19.2 < sci.maglimit < 22.
        c4 = sci.infobits == 0
        if c1 and c2 and c3 and c4:
            ok.append(sci)

    # get the very best images
    top = sorted(ok, key=lambda i: i.maglimit, reverse=True)[:50]
    if len(top) == 0:
        print(f'Not enough images ({len(top)} < 14) to make reference '
              f'{coaddname}. Skipping...')
        db.DBSession().rollback()
        continue

    
    coaddname = os.path.join(d, f'ref.{ok[0].field:06d}_c{ok[0].ccdid:02d}'
                                f'_q{ok[0].qid}_{fmap[ok[0].fid]}.{version}.fits')

    if len(top) < 14:
        print(f'Not enough images ({len(top)} < 14) to make reference '
              f'{coaddname}. Skipping...')
        db.DBSession().rollback()
        continue


    try:
        coadd = db.ReferenceImage.from_images(top, coaddname,
                                              data_product=True,
                                              nthreads=mpi.get_nthreads(),
                                              tmpdir='./tmp')
        coadd.version = version
    except TypeError as e:
        print(e, [t.basename for t in top], coaddname)
        db.DBSession().rollback()
        continue
    else:
        db.DBSession().add(coadd)
        catalog = db.PipelineFITSCatalog.from_image(coadd)
        catcopy = db.HTTPArchiveCopy.from_product(catalog)
        archive.archive(catcopy)
        db.DBSession().add(catcopy)
        db.DBSession().commit()

    t_stop = time.time()
    print(f'it took {t_stop - t_start} sec to make {coaddname}.', flush=True)

