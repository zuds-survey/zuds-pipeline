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
sources = mpi.get_my_share_of_work(infile)

for source_id in sources:

    tstart = time.time()
    sstart = time.time()
    source = db.DBSession().query(db.models.Source).get(source_id)

    bestdet = source.best_detection
    best_sub = bestdet.image
    best_new = best_sub.target_image
    best_ref = best_sub.reference_image

    field = f'{best_sub.field:06d}'
    ccdid = f'c{best_sub.ccdid:02d}'
    qid = f'q{best_sub.qid}'
    fid = f'{fmap[best_sub.fid]}'

    stamps = []

    for i, image in enumerate([best_sub, best_new, best_ref]):
        path = f'/global/cscratch1/sd/dgold/zuds/{field}/{ccdid}/{qid}/' \
               f'{fid}'
        image.find_in_dir(path)
        image.mask_image.find_in_dir(path)

    try:
        remapped = best_ref.aligned_to(best_sub)
        stamps = []
        for i in [best_sub, best_new, remapped]:
            # make a stamp for the first detection
            stamp = db.Thumbnail.from_detection(bestdet, i)
            stamps.append(stamp)
        source.add_linked_thumbnails()
    except Exception as e:
        print(e, flush=True)
        db.DBSession().rollback()
        continue

    db.DBSession().add_all(stamps)
    db.DBSession().commit()
