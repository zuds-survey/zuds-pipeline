import db
import os
import sys
import mpi
import pandas as pd
from pathlib import Path

fmap = {1: 'zg',
        2: 'zr',
        3: 'zi'}

db.init_db()

__author__ = 'Danny Goldstein <danny@caltech.edu>'
__whatami__ = 'Make the references for ZUDS.'

infile = sys.argv[1]  # file listing all the directories to build refs for
min_date = pd.to_datetime(sys.argv[2])  # minimum allowable date for refimgs
max_date = pd.to_datetime(sys.argv[3])  # maximum allowable date for refimgs

# get the work
my_dirs = mpi.get_my_share_of_work(infile)

# make a reference for each directory
for d in my_dirs:

    # get all the science images
    image_objects = []
    sci_fns = Path(d).glob('ztf*sciimg.fits')

    # load the objects partway into memory
    ok = []
    for fn in sci_fns:
        sci = db.ScienceImage.from_file(fn)
        c1 = min_date <= sci.obsdate <= max_date
        c2 = 1.7 < sci.seeing < 2.5
        c3 = sci.maglimit > 19.5
        if c1 and c2 and c3:
            ok.append(sci)

    # get the very best images
    top = sorted(ok, key=lambda i: i.maglimit, reverse=True)[:50]
    coaddname = os.path.join(d, f'ref.{ok[0].field:06d}_c{ok[0].ccdid:02d}'
                                f'_q{ok[0].qid}_{fmap[ok[0].fid]}.fits')
    try:
        coadd = db.ReferenceImage.from_images(top, coaddname,
                                              data_product=True,
                                              nthreads=64,
                                              tmpdir='./tmp')
    except TypeError as e:
        print(e, [t.basename for t in top], coaddname)
        db.DBSession().rollback()
        continue

    pwd = os.getcwd()
    os.chdir(d)
    coadd.rms_image.save()
    os.chdir(pwd)
    db.DBSession().add(coadd)
    db.DBSession().commit()







