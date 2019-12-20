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
__whatami__ = 'Make the subtractions for ZUDS.'

infile = sys.argv[1]  # file listing all the images to make subtractions of
refvers = sys.argv[2]

#subclass = db.MultiEpochSubtraction
#sciclass = db.ScienceCoadd

subclass = db.SingleEpochSubtraction
sciclass = db.ScienceImage


# get the work
imgs = mpi.get_my_share_of_work(infile)

# make a reference for each directory
for fn in imgs:
    tstart = time.time()

    sstart = time.time()
    sci = sciclass.from_file(fn)
    sstop = time.time()
    print(
        f'sci: {sstop-sstart:.2f} sec to load  {sci.basename}',
        flush=True
    )

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

    rstart = time.time()
    ref = db.ReferenceImage.from_file(refname, use_existing_record=True)
    rstop = time.time()

    print(
        f'ref: {rstop-rstart:.2f} sec to load ref for {sci.basename}',
        flush=True
    )


    basename = db.sub_name(sci.basename, ref.basename)

    prev = subclass.get_by_basename(basename)
    #prev=None

    #if (prev is not None) and (prev.modified is not None) and \
    #   (prev.modified > datetime.now() - timedelta(hours=24)):
    #    db.DBSession().rollback()
    #    continue

    if prev is not None:
        continue

    substart = time.time()
    try:
        sub = subclass.from_images(sci, ref,
                                   data_product=False,
                                   tmpdir='tmp')
    except Exception as e:
        print(e, [sci.basename, ref.basename], flush=True)
        db.DBSession().rollback()
        continue

    substop = time.time()
    print(
        f'sub: {substop-substart:.2f} sec to make {sub.basename}',
        flush=True
    )


    catstart = time.time()
    try:
        cat = db.PipelineFITSCatalog.from_image(sub)
    except Exception as e:
        print(e, [sub.basename], flush=True)
        db.DBSession.rollback()
        continue
    catstop = time.time()
    print(
        f'cat: {catstop-catstart:.2f} sec to make catalog for {sub.basename}',
        flush=True
    )

    dstart = time.time()
    try:
        detections = db.Detection.from_catalog(cat, filter=True)
    except Exception as e:
        print(e, [cat.basename], flush=True)
        db.DBSession.rollback()
        continue

    if len(detections) > 50:
        db.DBSession().rollback()
        print(f'{len(detections)} detections on "{sub.basename}", '
              'something wrong with the image probably', flush=True)

    dstop = time.time()
    print(
        f'det: {dstop-dstart:.2f} sec to make detections for {sub.basename}',
        flush=True
    )

    stampstart = time.time()
    try:
        if isinstance(sub, db.SingleEpochSubtraction):
            sub_target = sub.aligned_to(sub.reference_image)
        else:
            sub_target = sub
        if isinstance(sub.target_image, db.ScienceImage):
            new_target = sub.target_image.aligned_to(sub.reference_image)
        else:
            new_target = sub.target_image
        stamps = []
        for detection in detections:
            if len(detection.source.thumbnails) == 0:
                for i, t in zip(
                    [sub_target, new_target, sub.reference_image],
                    ['sub', 'new', 'ref']
                ):
                    # make a stamp for the first detection
                    stamp = db.models.Thumbnail.from_detection(detection, i)
                    stamp.type = t
                    stamps.append(stamp)
                thumbs = detection.source.return_linked_thumbnails()
                for thumb in thumbs:
                    thumb.source = detection.source
                stamps.extend(thumbs)
    except Exception as e:
        print(e, [cat.basename], flush=True)
        db.DBSession.rollback()
        continue

    stampstop = time.time()
    print(
        f'stamp: {stampstop-stampstart:.2f} sec to make stamps for {sub.basename}',
        flush=True
    )

    # make the alerts


    archstart = time.time()
    #subcopy = db.HTTPArchiveCopy.from_product(sub)
    #catcopy = db.HTTPArchiveCopy.from_product(cat)
    #mskcopy = db.HTTPArchiveCopy.from_product(sub.mask_image)
    db.DBSession().add_all(detections)
    #db.DBSession().add_all(stamps)c

    #db.DBSession().add(mskcopy)
    #db.DBSession().add(catcopy)
    #db.DBSession().add(subcopy)
    #archive.archive(subcopy)
    #archive.archive(catcopy)
    #archive.archive(mskcopy)
    db.DBSession().commit()
    archstop = time.time()
    print(
        f'archive: {archstop-archstart:.2f} sec to archive stuff for '
        f'{sub.basename}',
        flush=True
    )


    cleanstart = time.time()
    targets = []
    if hasattr(sci, '_rmsimg'):
        targets.append(sci.rms_image.local_path)
    if hasattr(sci, '_weightimg'):
        targets.append(sci.weight_image.local_path)

    sci.unmap()
    for target in targets:
        os.remove(target)

    cleanstop = time.time()

    tstop = time.time()
    print(f'clean: took {cleanstop - cleanstart} sec to clean '
          f'up after {sub.basename}"',
          flush=True)
    print(f'took {tstop - tstart} sec to make "{sub.basename}"', flush=True)
