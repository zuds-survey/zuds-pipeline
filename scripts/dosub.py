import db
import sys
import mpi
import os
import time
import archive
from datetime import datetime, timedelta

fmap = {1: 'zg',
        2: 'zr',
        3: 'zi'}

db.init_db()
#db.DBSession().autoflush = False
#db.DBSession().get_bind().echo = True

__author__ = 'Danny Goldstein <danny@caltech.edu>'
__whatami__ = 'Make the references for ZUDS.'

infile = sys.argv[1]  # file listing all the images to make subtractions of
refvers = sys.argv[2]

# get the work
imgs = mpi.get_my_share_of_work(infile)

# make a reference for each directory
for fn in imgs:
    tstart = time.time()
    sci = db.ScienceImage.from_file(fn)
    field = f'{sci.field:06d}'
    ccdid = f'c{sci.ccdid:02d}'
    qid = f'q{sci.qid}'
    fid = f'{fmap[sci.fid]}'
    refname = f'/global/cscratch1/sd/dgold/zuds/{field}/{ccdid}/{qid}/' \
              f'{fid}/ref.{field}_{ccdid}_{qid}_{fid}.{refvers}.fits'

    if not (db.ReferenceImage.get_by_basename(os.path.basename(refname))
            and os.path.exists(refname)):
        print(f'Ref {refname} does not exist. Skipping...')
        continue

    ref = db.ReferenceImage.from_file(refname, use_existing_record=True,
                                      load_header=False)

    basename = db.sub_name(sci.basename, ref.basename)
    #prev = db.SingleEpochSubtraction.get_by_basename(basename)
    prev=None

    if (prev is not None) and (prev.modified is not None) and \
       (prev.modified > datetime.now() - timedelta(hours=15)):
        db.DBSession().rollback()
        continue


    try:
        sub = db.SingleEpochSubtraction.from_images(sci, ref,
                                                    data_product=False,
                                                    tmpdir='tmp')
    except Exception as e:
        print(e, [sci.basename, ref.basename], flush=True)
        db.DBSession().rollback()
        continue

    try:
        cat = db.PipelineFITSCatalog.from_image(sub)
    except Exception as e:
        print(e, [sub.basename], flush=True)
        db.DBSession.rollback()
        continue

    dstart = time.time()
    try:
        detections = db.Detection.from_catalog(cat, filter=True)
    except Exception as e:
        print(e, [cat.basename], flush=True)
        db.DBSession.rollback()
        continue
    dstop = time.time()
    print(
        f'det: {dstop-dstart:.2f} sec to make detections for {sub.basename}',
        flush=True
    )

    try:
        remapped = sub.reference_image.aligned_to(sub)
        stamps = []
        for i in [sub, sub.target_image, remapped]:
            for detection in detections:
                stamp = db.Stamp.from_detection(detection, i)
                stamps.append(stamp)
    except Exception as e:
        print(e, [cat.basename], flush=True)
        db.DBSession.rollback()
        continue

    subcopy = db.HTTPArchiveCopy.from_product(sub)
    catcopy = db.HTTPArchiveCopy.from_product(cat)
    mskcopy = db.HTTPArchiveCopy.from_product(sub.mask_image)
    db.DBSession().add_all(detections)
    db.DBSession().add_all(stamps)

    db.DBSession().add(mskcopy)
    db.DBSession().add(catcopy)
    db.DBSession().add(subcopy)
    archive.archive(subcopy)
    archive.archive(catcopy)
    archive.archive(mskcopy)

    db.DBSession().commit()
    tstop = time.time()

    print(f'took {tstop - tstart} sec to make "{sub.basename}"', flush=True)
