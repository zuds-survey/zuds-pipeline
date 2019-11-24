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

    # find the reference
    copy = db.DBSession().query(db.HTTPArchiveCopy)\
        .join(db.ReferenceImage,
              db.ReferenceImage.id == db.HTTPArchiveCopy.product_id) \
        .filter(
        db.ReferenceImage.field == sci.field,
        db.ReferenceImage.ccdid == sci.ccdid,
        db.ReferenceImage.qid == sci.qid,
        db.ReferenceImage.fid == sci.fid,
    ).options(db.sa.joinedload('copies')).first()

    ref = db.DBSession().from_file(copy.archive_path)

    try:
        sub = db.SingleEpochSubtraction.from_images(sci, ref, data_product=True)
    except Exception as e:
        print(e, [sci.basename, ref.basename])
        db.DBSession().rollback()
        continue

    db.DBSession().add(sub)
    db.DBSession().commit()









