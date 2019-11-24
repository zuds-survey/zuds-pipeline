import db
import sys
import mpi
import archive

fmap = {1: 'zg',
        2: 'zr',
        3: 'zi'}

db.init_db()

__author__ = 'Danny Goldstein <danny@caltech.edu>'
__whatami__ = 'Make the references for ZUDS.'

infile = sys.argv[0]  # file listing all the images to make subtractions of

# get the work
imgs = mpi.get_my_share_of_work(infile)

# make a reference for each directory
for fn in imgs:
    sci = db.ScienceImage.from_file(fn)
    field = f'{sci.field:06d}'
    ccdid = f'c{sci.ccdid:02d}'
    qid = f'q{sci.qid}'
    fid = f'{fmap[sci.fid]}'
    ref = db.DBSession().from_file(f'/global/cscratch1/sd/dgold/zuds/{field}/{ccdid}/{qid}/{fid}/ref.{field}_{ccdid}_{qid}_{fid}.fits')

    try:
        sub = db.SingleEpochSubtraction.from_images(sci, ref, data_product=True)
    except Exception as e:
        print(e, [sci.basename, ref.basename])
        db.DBSession().rollback()
        continue

    db.DBSession().add(sub)
    db.DBSession().commit()









