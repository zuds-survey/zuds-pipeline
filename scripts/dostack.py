import db
import sys
import mpi
import os
import time
import archive
import pandas as pd

fmap = {1: 'zg',
        2: 'zr',
        3: 'zi'}

db.init_db()
db.DBSession().get_bind().echo = True

__author__ = 'Danny Goldstein <danny@caltech.edu>'
__whatami__ = 'Make the references for ZUDS.'

infile = sys.argv[1]  # file listing all the images to make subtractions of

# get the work
jobs = mpi.get_my_share_of_work(infile, reader=pd.read_csv)


for _, job in jobs.iterrows():

    tstart = time.time()
    images = db.DBSession().query(db.ZTFFile).filter(
        db.ZTFFile.id.in_(job['target'].tolist())
    ).all()
    db.ensure_images_have_the_same_properties(images)

    field = f'{images[0].field:06d}'
    ccdid = f'c{images[0].ccdid:02d}'
    qid = f'q{images[0].qid}'
    fid = f'{fmap[images[0].fid]}'

    for image in images:
        path = f'/global/cscratch1/sd/dgold/zuds/{field}/{ccdid}/{qid}/' \
               f'{fid}/{image.basename}'
        image.map_to_local_file(path)
        image.mask_image.map_to_local_file(path.replace('.fits', '.mask.fits'))

    basename = f'sub.{field}_{ccdid}_{qid}_{fid}_{job["binleft"]}_' \
               f'{job["binright"]}.coadd.fits'
    prev = db.ScienceCoadd.get_by_basename(basename)
    outname = os.path.join(os.path.dirname(images[0].local_path), basename)

    if prev is not None:
        continue

    try:
        sub = db.ScienceCoadd.from_images(images, outname,
                                          nthreads=mpi.get_nthreads(),
                                          data_product=False,
                                          tmpdir='tmp')
    except Exception as e:
        print(e, [i.basename for i in images])
        db.DBSession().rollback()
        continue

    try:
        catalog = db.PipelineFITSCatalog.from_image(sub)
    except Exception as e:
        print(e, [i.basename for i in images])
        db.DBSession().rollback()
        continue

    sub.binleft = pd.to_datetime(job['binleft'])
    sub.binright = pd.to_datetime(job['binright'])

    db.DBSession().add(sub)
    db.DBSession().add(catalog)

    db.DBSession().rollback()
    tstop = time.time()

    print(f'took {tstop - tstart} sec to make "{sub.basename}"')










