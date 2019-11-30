import db
import sys
import mpi
import os
import time
import archive

fmap = {1: 'zg',
        2: 'zr',
        3: 'zi'}

db.init_db()
db.DBSession().get_bind().echo = True

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

    if not os.path.exists(refname):
        print(f'Ref {refname} does not exist. Skipping...')
        continue

    ref = db.ReferenceImage.from_file(refname, use_existing_record=True)

    try:
        sub = db.SingleEpochSubtraction.from_images(sci, ref,
                                                    data_product=False,
                                                    tmpdir='tmp')
    except Exception as e:
        print(e, [sci.basename, ref.basename])
        db.DBSession().rollback()
        continue

    db.DBSession().add(sub)
    db.DBSession().rollback()
    tstop = time.time()

    print(f'took {tstop - tstart} sec to make "{sub.basename}"')










